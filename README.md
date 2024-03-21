# PromptExplainer

**PromptExplainer: Explaining Language Models through Prompt-based Learning (EACL 2024)** [Link](https://aclanthology.org/2024.findings-eacl.60/)


PromptExplainer explores using token distributions to explain masked language models such as BERT and RoBERTa. Our another work, TDD, utilizes token distributions to explain autoregressive LLMs. Welcome to check [TDD](https://github.com/zijian678/TDD)!

## Implementations

We provide two implementations:
* Simple implementation: [PromptExplainer.ipynb](https://github.com/zijian678/PromptExplainer/blob/main/PromptExplainer.ipynb). Please note that this is just a simple implementation of PromptExplainer to help understand our framework more easily. For experimental comparison purposes, please follow our original paper to select the templates and verbalizers.
* Integration into existing prompt-based learning frameworks. Taking the [OpenPrompt](https://github.com/thunlp/OpenPrompt) that provides available codes for multiple SOTA prompt-based models as an example, PromptExplainer can be implemented with a few lines of work by modifying the pipline_base.py. The **forward** function in the PromptForClassification - pipeline_base.py can be modified as follwos. E is the obtained input saliency.

```
        outputs = self.prompt_model(batch)  # eqn 1
        outputs = self.verbalizer.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        # PromptExplainer
        os1, os2, os3 = outputs.shape
        output_all = outputs.reshape(os1 * os2, os3)
        all_token_logits = self.verbalizer.process_outputs(output_all,
                                                           batch=batch)  # eqn 2
        all_token_logits = all_token_logits.reshape(os1, os2, label_words_logits.shape[
            1])
        all_token_softmax = F.softmax(all_token_logits, dim=-1) # eqn 3
        E = all_token_softmax[:,:,i] # eqn 4, i is the class id
```
## Reproduce

Our results can be reproduced using [OpenPrompt](https://github.com/thunlp/OpenPrompt) and [KPT](https://github.com/thunlp/KnowledgeablePromptTuning). For the activation and pruning tasks, please follow [
XAI_Transformers](https://github.com/AmeenAli/XAI_Transformers). We express our heartfelt gratitude to the authors for their outstanding contributions!

## Citation
If you find our work useful, please consider citing PromptExplainer:

```
@inproceedings{feng-etal-2024-promptexplainer,
    title = "{P}rompt{E}xplainer: Explaining Language Models through Prompt-based Learning",
    author = "Feng, Zijian  and
      Zhou, Hanzhang  and
      Zhu, Zixiao  and
      Mao, Kezhi",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.60",
    pages = "882--895",
}

```
