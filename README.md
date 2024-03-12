# PromptExplainer

PromptExplainer: Explaining Language Models through Prompt-based Learning (EACL 2024)

## Implementations

We provide two implementations:
* Simple implementation: [PromptExplainer.ipynb](https://github.com/zijian678/PromptExplainer/blob/main/PromptExplainer.ipynb). Please note that this is just a simple implementation of PromptExplainer to help understand our framework more easily. For experimental comparison purposes, please follow our original paper to select the templates and verbalizers.
* Integration into existing prompt-based learning frameworks. Taking the [OpenPrompt](https://github.com/thunlp/OpenPrompt) that provides available codes for multiple SOTA prompt-based models as an example, PromptExplainer can be implemented with a few lines of work by modifying the pipline_base.py. The **forward** function in the PromptForClassification - pipeline_base.py can be modified as:

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
