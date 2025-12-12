/**
 * @brief Max pooling layer stub.
 */
#pragma once

#include <stdexcept>

#include "ml/conv_layer/interface.h"
#include "ml/utils.h"
#include "ml/types.h"

namespace ml::conv_layer::stub
{
/**
 * @brief Max pooling layer stub.
 */
class MaxPool final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] poolSize Pool size. Must divide the input size.
     */
    explicit MaxPool(const std::size_t inputSize, const std::size_t poolSize)
        : myInput{}
        , myInputGradients{}
        , myOutput{}
    {
        // Check the pool dimensions, throw an exception if invalid.
        if (0U == inputSize)
        {
            throw std::invalid_argument("Input size cannot be 0!");
        }
        else if (0U == poolSize)
        {
            throw std::invalid_argument("Pool size cannot be 0!");   
        }
        else if (inputSize < poolSize)
        {
            throw std::invalid_argument("Input size cannot be smaller than the pool size!");
        }
        else if (0 != (inputSize % poolSize))
        {
            throw std::invalid_argument("Input size must be divisible by pool size!");
        }

        // Initialize the pool matrices.
        const std::size_t outputSize{inputSize / poolSize};
        initMatrix(myInput, inputSize);
        initMatrix(myInputGradients, inputSize);
        initMatrix(myOutput, outputSize);
    }

    /**
     * @brief Default destructor.
     */
    ~MaxPool() noexcept override = default;

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
        constexpr const char* opName{"feedforward in max pooling layer"};
        return matchDimensions(myInput.size(), input.size(), opName)
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
        constexpr const char* opName{"backpropagation in max pooling layer"};
        return matchDimensions(myOutput.size(), outputGradients.size(), opName)
            && isMatrixSquare(outputGradients, opName);
    }

    /**
     * @brief Perform optimization (not implemented for pooling layers).
     * 
     * @param[in] learningRate Learning rate to use.
     * 
     * @return True (optimization is a no-op for pooling layers).
     */
    bool optimize(const double learningRate) noexcept override
    {
        (void) (learningRate);
        return true;
    }

    MaxPool()                          = delete; // No default constructor.
    MaxPool(const MaxPool&)            = delete; // No copy constructor.
    MaxPool(MaxPool&&)                 = delete; // No move constructor.
    MaxPool& operator=(const MaxPool&) = delete; // No copy assignment.
    MaxPool& operator=(MaxPool&&)      = delete; // No move assignment.

private:
    /** Pool input. */
    Matrix2d myInput;

    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Pool output. */
    Matrix2d myOutput;
};
} // namespace ml::conv_layer::stub
