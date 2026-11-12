### Fine Tuning Techniques
- Utalize Lora back propogation techniques as discussed in class - freeze current weights and only update small adapater
    - Forward propogation - Take x input and compare it to y input: y predicted = weight times the input
    - Loss Function - use sometype of distance formula to calculate loss(square-loss/manhattan): 1/2(y^ - y)^2
    - Back propogation - Using chain rules find which weight contributed to error: derivate of L / derivative of W
    - Gradient Descent - Update model weights, loss function, and y predicted
- By making small prebutions we can make the model perform better under different conditions.

### Defending against adverial attacks
- Adversial attacks are done by prompt injections, jailbreak attacks and more
    - Create a loop for llms to generate some attacks and train you llm to defend against it.
    - Test cases, prompt inject
    - Classify whether or not a prompt is malicious before returning the input