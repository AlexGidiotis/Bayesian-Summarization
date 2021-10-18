import logging
import os
import shutil
import time
import json
import random
import linecache
import gc

import numpy as np

import torch

from src.bayesian_summarization.bayesian import BayesianSummarizer
from src.common.loaders import load_model, create_loader
from src.bayesian_summarization.bleu import analyze_generation_bleuvar
from src.summarization.sum_base import Summarizer


logger = logging.getLogger(__name__)


def write_metrics(out_path, metrics, filename):
    with open(os.path.join(out_path, f"{filename}.json"), "a+") as hist:
        json.dump(metrics, hist)
        hist.write("\n")


class ActiveSum:
    """
    Base class for active summarization training.
    Implements the basic training step and can be extended
    with other active learning strategies.
    """
    def __init__(self, data_sampler, device, **kwargs):
        self.data_sampler = data_sampler
        self.device = device
        self.metric = None
        self.save_limit = None
        self.save_step = None
        self.beams = None
        self.source_len = None
        self.target_len = None
        self.val_samples = None
        self.test_samples = None
        self.lr = None
        self.batch_size = None
        self.seed = None
        self.sum_col = None
        self.doc_col = None
        self.init_model = None
        self.py_module = None
        self.best_score = 0.
        self.__dict__.update(kwargs)

    def train(
            self,
            labeled_path,
            model_path,
            eval_path,
            epochs,
    ):
        """
        Wraps the call to the summarization trainer with the required arguments
        """
        train_path = os.path.join(labeled_path, "train.json")
        sum_trainer = Summarizer(
            model_name_or_path=self.init_model,
            tokenizer_name=self.init_model,
            train_file=train_path,
            validation_file=eval_path,
            text_column=self.doc_col,
            summary_column=self.sum_col,
            output_dir=model_path,
            logging_dir=f"{model_path}/logs",
            seed=self.seed,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            overwrite_output_dir=True,
            max_val_samples=self.val_samples,
            max_test_samples=self.test_samples,
            learning_rate=self.lr,
            adafactor=True,
            max_source_length=self.source_len,
            max_target_length=self.target_len,
            val_max_target_length=self.target_len,
            pad_to_max_length=True,
            num_beams=self.beams,
            num_train_epochs=epochs,
            save_steps=self.save_step,
            save_total_limit=self.save_limit,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            metric_for_best_model=self.metric,
            greater_is_better=True,
            do_train=True,
            do_eval=True,
            predict_with_generate=True,
            do_predict=False,
        )

        sum_trainer.init_sum()
        train_metrics = sum_trainer.train()
        eval_metrics = sum_trainer.evaluate()

        del sum_trainer
        gc.collect()
        torch.cuda.empty_cache()

        return train_metrics, eval_metrics

    def train_step(self, labeled_path, model_path, eval_path, epochs):
        """Runs the basic training step and evaluates metrics.

        Also, keeps track of the best model based on the primary metric
        and does checkpointing and metrics logging.
        """
        train_metrics, eval_metrics = self.train(
            labeled_path=labeled_path,
            model_path=model_path,
            eval_path=eval_path,
            epochs=epochs,)

        write_metrics(model_path, train_metrics, filename="train_metrics_hist")
        write_metrics(model_path, eval_metrics, filename="eval_metrics_hist")

        if eval_metrics[f"eval_{self.metric}"] > self.best_score:
            best_checkpoint_path = os.path.join(model_path, "best_checkpoint")
            if not os.path.exists(best_checkpoint_path):
                os.mkdir(best_checkpoint_path)

            shutil.copy(os.path.join(model_path, "pytorch_model.bin"), best_checkpoint_path)
            self.best_score = eval_metrics[f"eval_{self.metric}"]

            logger.info(f"Best model with {self.metric} score {self.best_score} saved to {best_checkpoint_path}")

        linecache.clearcache()

    def predict(self):
        pass

    def obtain_targets(self):
        pass

    def write_samples(self, sample, selected_samples, sample_idxs, out_path, scores=None):
        """Writes a dataset sample to be used for training.

        Three files are written:
        1) train.json: with the training (input, target) pairs, one per line
        2) metadata.json: full sample metadata including ranking score
        3) all_scores.json: all ranking scores computed in the step (includes examples that weren't selected)
        """
        metadata_output = os.path.join(out_path, "metadata.json")
        train_output = os.path.join(out_path, "train.json")
        score_stats = os.path.join(out_path, "all_scores.json")
        mdf = open(metadata_output, "a+")
        outf = open(train_output, "a+")
        entf = open(score_stats, "a+")
        for di, (data_s, data_idx, si) in enumerate(zip(sample, sample_idxs, selected_samples)):
            out_json = {"sample_id": data_idx, "score": scores[si] if scores is not None else None, "sample": data_s}
            json.dump(out_json, mdf)
            mdf.write("\n")

            train_sample_json = {
                "document": data_s[self.doc_col].replace('\n', ' '),
                "summary": data_s[self.sum_col].replace('\n', ' '),
                "id": data_idx}
            json.dump(train_sample_json, outf)
            outf.write("\n")

        json.dump({"all_scores": scores}, entf)
        entf.write("\n")

        mdf.close()
        outf.close()
        entf.close()

    def init_learner(self, init_labeled, model_path, labeled_path, eval_path, epochs):
        """Initialize the active learner.

        The initial model is created and trained on an pool of randomly selected examples.
        """
        sample, sample_idxs = self.data_sampler.sample_data(k=init_labeled)
        self.data_sampler.remove_samples(sample_idxs)

        self.write_samples(sample, sample_idxs, sample_idxs, labeled_path)
        train_metrics, eval_metrics = self.train(
            labeled_path=labeled_path,
            model_path=model_path,
            eval_path=eval_path,
            epochs=epochs,)

        logger.info("Finished initial training")
        self.best_score = eval_metrics[f"eval_{self.metric}"]

        write_metrics(model_path, eval_metrics, filename="eval_metrics_hist")

        linecache.clearcache()

    def resume_learner(self, labeled_path):
        """
        """
        # load previously selected samples
        metadata_output = os.path.join(labeled_path, "metadata.json")
        selected_idxs = []
        with open(metadata_output) as mf:
            for s in mf:
                selected_idxs.append(json.loads(s.strip())["sample_id"])

        # remove samples from data_sampler
        self.data_sampler.remove_samples(selected_idxs)
        logger.info(f"{len(selected_idxs)} have been removed from the pool")


class BAS(ActiveSum):
    """
    Bayesian active summarization module.

    At each learning step we sample summaries with MC dropout for a pool
    of unlabelled examples. Then we used the BLEUVar metric computed based
    on the sampled summaries to select a subset to be labelled.

    The labelled set L is extended with the selected subset at each step
    and a new model is trained on the extended labelled set.

    Example:
    ```
    active_learner = BAS(
        train_sampler,
        device='gpu',
        doc_col='document',
        sum_col='summary,
        seed=100,
        py_module='as_learn.py',
        init_model='google/pegasus-large',
        source_len=128,
        target_len=16,
        val_samples=100,
        batch_size=6,
        beams=4,
        lr=1e-4,
        save_step=100,
        save_limit=1,
        metric='rouge_1',
    )

    active_learner.init_learner(
        init_labeled=20,
        model_path='path_to_model/model',
        labeled_path='path_to_data',
        eval_path='path_to_data/validation.json',
        epochs=10)

    active_learner.learn(
        steps=10,
        model_path='path_to_model/model',
        labeled_path='path_to_data',
        k=100, s=10, n=10,
        eval_path='path_to_data/validation.json',
        epochs=10)
    ```
    """
    def __init__(self, data_sampler, device, **kwargs):
        super(BAS, self).__init__(data_sampler, device, **kwargs)

    def learn(self, steps, model_path, labeled_path, k, s, n, eval_path, epochs):
        """Learning strategy for Bayesian summarization"""
        for i in range(steps):
            self.learning_step(model_path, labeled_path, k, s, n, eval_path=eval_path, epochs=epochs, step=i)

    def mc_sample(self, sample_idxs, model_path, n):
        """Sample summaries with MC dropout

        Summaries are sampled for a list of document ids (sample_idxs) from self.data_sampler.
        For each input we run N stochastic forward passes with dropout enabled in order to get
        n different summaries.
        """
        dataloader = create_loader(self.data_sampler.dataset, batch_size=self.batch_size, sample=sample_idxs)

        model, tokenizer = load_model(tokenizer_name=self.init_model, model_name_or_path=model_path)
        model = model.to(self.device)
        bayesian_summarizer = BayesianSummarizer(model=model, tokenizer=tokenizer)

        generated_sums = bayesian_summarizer.generate_mc_summaries(
            dataloader,
            device=self.device,
            text_column=self.doc_col,
            max_source_length=self.source_len,
            num_beams=self.beams,
            n=n)

        del bayesian_summarizer, tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()

        return generated_sums

    def rank_data(self, mc_generations, n):
        """Compute the BLEUVar scores for summaries generated with MC dropout"""
        bleuvars = []
        for gen_list in mc_generations:
            bleuvar, _, _, _ = analyze_generation_bleuvar(gen_list, n)
            bleuvars.append(bleuvar)

        return bleuvars

    def select_data(self, scores, s, threshold=0.96):
        """Select the examples with the S highest BLEUVar scores"""
        scores = np.array(scores)
#         top_s = scores.argsort()[-s:][::-1]
        top_s = scores.argsort()
        top_s = [idx for idx in top_s if scores[idx] < threshold][-s:][::-1]

        np.random.shuffle(top_s)
        top_s_scores = [scores[i] for i in top_s]

        return top_s, top_s_scores

    def learning_step(self, model_path, labeled_path, k, s, n, eval_path, epochs, step=0):
        """A single learning step for Bayesian active summarization

        1) select a pool of unlabeled examples
        2) sample N summaries with MC dropout for the examples in the selected pool
        3) compute BLEUVar scores for the selected pool
        4) add the S examples with the highest BLEUVar to the labelled set
        5) train on the extended labelled set
        """
        logger.info(f"Learning step {step}")
        si_time = time.time()
        sample, sample_idxs = self.data_sampler.sample_data(k=k)
        mc_gens = self.mc_sample(sample_idxs, model_path, n)
        unc_scores = self.rank_data(mc_gens, n)
        selected_samples, selected_scores = self.select_data(unc_scores, s)
        selected_idxs = [sample_idxs[i] for i in selected_samples]

        self.data_sampler.remove_samples(selected_idxs)
        self.write_samples(
            sample.select(selected_samples),
            selected_samples,
            selected_idxs,
            labeled_path,
            scores=unc_scores)

        self.train_step(labeled_path, model_path, eval_path, epochs)
        ei_time = time.time()
        logger.info(f"Finished learning step {step}: {ei_time - si_time} sec.")


class RandomActiveSum(ActiveSum):
    """
    Random active summarization module.

    At each learning step we randomly select a subset to be labelled.

    The labelled set L is extended with the selected subset at each step
    and a new model is trained on the extended labelled set.

    Example:
    ```
    active_learner = RandomActiveSum(
        train_sampler,
        device='gpu',
        doc_col='document',
        sum_col='summary,
        seed=100,
        py_module='as_learn.py',
        init_model='google/pegasus-large',
        source_len=128,
        target_len=16,
        val_samples=100,
        batch_size=6,
        beams=4,
        lr=1e-4,
        save_step=100,
        save_limit=1,
        metric='rouge_1',
    )

    active_learner.init_learner(
        init_labeled=20,
        model_path='path_to_model/model',
        labeled_path='path_to_data',
        eval_path='path_to_data/validation.json',
        epochs=10)

    active_learner.learn(
        steps=10,
        model_path='path_to_model/model',
        labeled_path='path_to_data',
        k=100, s=10,
        eval_path='path_to_data/validation.json',
        epochs=10)
    ```
    """
    def __init__(self, data_sampler, device, **kwargs):
        super(RandomActiveSum, self).__init__(data_sampler, device, **kwargs)

    def learn(self, steps, model_path, labeled_path, k, s, eval_path, epochs):
        """Learning strategy"""
        for i in range(steps):
            self.learning_step(model_path, labeled_path, k, s, eval_path=eval_path, epochs=epochs, step=i)

    def learning_step(self, model_path, labeled_path, k, s, eval_path, epochs, step=0):
        """A single learning step for active summarization

        1) select a pool of unlabeled examples
        2) randomly add S examples from the unlabelled pool to the labelled set
        5) train on the extended labelled set"""
        logger.info(f"Learning step {step}")
        si_time = time.time()
        sample, sample_idxs = self.data_sampler.sample_data(k=k)

        selected_idxs = sample_idxs[:s]
        selected_samples = list(range(s))

        self.data_sampler.remove_samples(selected_idxs)
        self.write_samples(sample.select(selected_samples), selected_samples, selected_idxs, labeled_path)

        self.train_step(labeled_path, model_path, eval_path, epochs)
        ei_time = time.time()
        logger.info(f"Finished learning step {step}: {ei_time - si_time} sec.")


class DataSampler:
    """Implements random sampling with and without replacement on a dataset

    Example:
    ```
    train_sampler = DataSampler(dataset, split="train")

    sample, sample_idxs = train_sampler.sample_data(k=100)
    selected_idxs = sample_idxs[:10]
    train_sampler.remove_samples(selected_idxs)
    remaining_sample_idxs = train_sampler.get_available_samples
    new_sample, new_sample_idxs = train_sampler.sample_data(k=100)  # removed samples cannot be resampled
    ```
    """
    def __init__(
            self,
            dataset,
            split,
    ):
        self.split = split
        self.dataset = dataset

        self.num_samples = self.dataset.info.splits[split].num_examples
        self.removed = []

    def sample_data(self, k):
        """Randomly select K samples from the remaining dataset"""
        available_samples = self.get_available_samples()
        random.shuffle(available_samples)
        sampled_idxs = available_samples[:k]
        sample_data = self.dataset.select(sampled_idxs)

        return sample_data, sampled_idxs

    def remove_samples(self, samples):
        """Remove a subset of samples (the removed can no longer be sampled)"""
        self.removed += samples

    def get_available_samples(self):
        """Get the available samples excluding samples that have been removed"""
        return [si for si in range(0, self.num_samples - 1) if si not in self.removed]