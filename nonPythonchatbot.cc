#include <torch/script.h>  // LibTorch API
#include <iostream>
#include <vector>

int main() {
    // Load the TorchScript model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("scripted_chatbot.pt");  // Load the saved model
        model.to(at::kCUDA);  // Move to GPU if available
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Chatbot Model Loaded Successfully!\n";

    // Example input tensor (dummy input, replace with actual tokenized input)
    std::vector<int64_t> input_tokens = {1, 42, 58, 99}; // Example tokenized sentence
    at::Tensor input_tensor = torch::tensor(input_tokens).unsqueeze(0).to(at::kCUDA);

    // Perform inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    at::Tensor output;
    try {
        output = model.forward(inputs).toTensor();
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during model inference: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Chatbot Response (Token IDs): " << output << "\n";

    return 0;
}
