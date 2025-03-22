#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    // Load the TorchScript model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("scripted_chatbot_cpu.pth");

        // Force the model to run on CPU
        model.to(torch::kCPU);
        model.eval();
        std::cout << "TorchScript Model Loaded Successfully!" << std::endl;

        // Example input - A batch of 4 sentences (tokenized)
        std::vector<std::vector<int64_t>> input_ids = {
            {1, 45, 230, 786},
            {1, 23, 67, 890},
            {2, 34, 99, 120},
            {3, 56, 111, 450}
        };

        // Convert input_ids into a 2D tensor and ensure it’s on CPU
        at::Tensor input_tensor = torch::stack({
            torch::tensor(input_ids[0], torch::kLong).to(torch::kCPU),
            torch::tensor(input_ids[1], torch::kLong).to(torch::kCPU),
            torch::tensor(input_ids[2], torch::kLong).to(torch::kCPU),
            torch::tensor(input_ids[3], torch::kLong).to(torch::kCPU)
        });

        // Ensure input_length is on CPU
        at::Tensor input_length = torch::tensor(
            { (int64_t)input_ids[0].size(),
              (int64_t)input_ids[1].size(),
              (int64_t)input_ids[2].size(),
              (int64_t)input_ids[3].size() }, 
            torch::kLong
        ).to(torch::kCPU);

        // Set max_length as an int (not a tensor)
        int max_length = 10;

        // Model forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        inputs.push_back(input_length);
        inputs.push_back(max_length);

        // Run inference
        at::Tensor output = model.forward(inputs).toTensor();
        std::cout << "Chatbot Response: " << output << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.msg() << std::endl;
        return -1;
    }

    return 0;
}

