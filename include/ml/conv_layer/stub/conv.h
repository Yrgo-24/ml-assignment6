/**
 * @brief Convolutional layer stub.
 */
#pragma once

#include <sstream>
#include <stdexcept>

#include "ml/act_func/type.h"
#include "ml/conv_layer/interface.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::conv_layer::stub
{
/**
 * @brief Convolutional layer stub.
 * 
 *        This class is non-copyable and non-movable.
 */
class Conv final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     * @param[in] actFunc Activation function to use (default = 0).
     */
    explicit Conv(const std::size_t inputSize, const std::size_t kernelSize, 
                  const act_func::Type actFunc = act_func::Type::None)
        : myInputGradients{}
        , myKernel{}
        , myOutput{}
    {
        // Throw exception if the kernel size is outside range [1, 11] or larger then the input size.
        if ((kMinKernelSize > kernelSize) || (kMaxKernelSize < kernelSize))
        {
            std::stringstream msg{};
            msg << "Invalid kernel size " << kernelSize << ": kernel size must be in range ["
                << kMinKernelSize << ", " << kMaxKernelSize << "]!\n";
            throw std::invalid_argument(msg.str());
        }
        else if (inputSize < kernelSize)
        {
            throw std::invalid_argument(
                "Failed to create convolutional layer: kernel size cannot be greater than input size!");
        }

        // Initialize the matrices with zeros.
        initMatrix(myInputGradients, inputSize);
        initMatrix(myKernel, kernelSize);
        initMatrix(myOutput, inputSize);

        // Ignore activation function in this implementation.
        (void) (actFunc);
    }

    /** 
     * @brief Destructor. 
     */
    ~Conv() noexcept override = default;

    /**
     * @brief Get the input size of the layer.
     * 
     * @return The input size of the layer.
     */
    std::size_t inputSize() const noexcept override { return myInputGradients.size(); }

    /**
     * @brief Get the output size of the layer.
     * 
     * @return The output size of the layer.
     */
    std::size_t outputSize() const noexcept override { return myOutput.size(); }

    /**
     * @brief Get the output of the layer.
     * 
     * @return Matrix holding the output of the layer.
     */
    const Matrix2d& output() const noexcept override { return myOutput; }

    /**
     * @brief Get the input gradients of the layer.
     * 
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override { return myInputGradients; }

    /**
     * @brief Perform feedforward operation.
     * 
     * @param[in] input Matrix holding input data.
     * 
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override
    {
        // Return true if the input matches the expected (unpadded) output size.
        constexpr const char* opName{"feedforward in convolutional layer"};
        return matchDimensions(myOutput.size(), input.size(), opName)
            && isMatrixSquare(input, opName);
    }

    /**
     * @brief Perform backpropagation.
     * 
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     * 
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix2d& outputGradients) noexcept override
    {
        // Return true if the output dimensions match.
        constexpr const char* opName{"backpropagation in convolutional layer"};
        return matchDimensions(myOutput.size(), outputGradients.size(), opName)
            && isMatrixSquare(outputGradients, opName);
    }

    /**
     * @brief Perform optimization.
     * 
     * @param[in] learningRate Learning rate to use.
     * 
     * @return True on success, false on failure.
     */
    bool optimize(const double learningRate) noexcept override
    {
        // Check the learning rate, return true if valid.
        constexpr const char* opName{"optimization in convolutional layer"};
        return checkLearningRate(learningRate, opName);
    }

    Conv()                       = delete; // No default constructor.
    Conv(const Conv&)            = delete; // No copy constructor.
    Conv(Conv&&)                 = delete; // No move constructor.
    Conv& operator=(const Conv&) = delete; // No copy assignment.
    Conv& operator=(Conv&&)      = delete; // No move assignment.

private:
    /** Minimum valid kernel size. */
    static constexpr std::size_t kMinKernelSize{1U};

    /** Minimum valid kernel size. */
    static constexpr std::size_t kMaxKernelSize{11U};

    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Kernel matrix. */
    Matrix2d myKernel;

    /** Output matrix. */
    Matrix2d myOutput;
};
} // namespace ml::conv_layer::stub
