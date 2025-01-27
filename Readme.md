# Using https://www.hailuo.ai/ to create this code base

## Testing Kimi vs DeepSeek vs Hailuo 

### Initial Prompt Used
Context: Build an MoE to be trained on wikipedia
Design: Neat and clean design to make sure MoE work on a context_length variable
Code: Pytorch
Data: Pytorch Datasets or Hugging face 
Modules: Model, visualization, logging, data, training/testing/evaluation
Documentation: Clean code comments, short and concise
Execution: Yaml settings file

Make sure we have 
- Data downloading, preprocessing, tokenization, correct context lengths and attention masking etc. 
- Visualization and logging are good
- Advanced training
- During validation, we provide a fixed input string/prompt against which text is generated
- File and console logging is very important
- Add a balance_loss so that the experts are balanced in a  way that they havve equal represenation


## Findings
As of today (27th Jan 2025) I tried DeepSeek at chat.deepseekcom, kimi at  https://kimi.moonshot.cn/ and  https://www.hailuo.ai/
- DeepSeek kept giving me short answers continously and missed visualization code a few times. Speed was good
- Kimi was the fastest model, acceptable code generated but also was minimalist in terms of giving comprehensive code
- hailuo Came out on top, I dont know the model name.  The UI and response was painfully slow but gave me code that was comprehensive. It seems like a reasoning model since it atleast in some occassions would generate code and revise it. All requests were fullfiled. Exceptions here and there.


## Conclusion
Coming from a ClaudeAI (Sonnet) heavy user experience, hailuo.ai definitely was better