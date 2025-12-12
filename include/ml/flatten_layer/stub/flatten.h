/**
 * @brief Flatten layer stub.
 */
#pragma once

#include <stdexcept>

#include "ml/flatten_layer/interface.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::flatten_layer::stub
{
/**s
 * @brief Flatten layer stub.
 * 
 *        This class is non-copyable and non-movable.
 */
class Flatten final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     */
    explicit Flatten(const std::size_t inputSize)
        : myInputGradients{}
        , myOutput{}
    {
        // Check the input size, throw an exception if invalid.
        if (0U == inputSize)
        {
            throw std::invalid_argument("Input size cannot be 0!");
        }

        // Initialize layer matrices.
        const std::size_t outputSize{inputSize * inputSize};
        initMatrix(myInputGradients, inputSize);
        initMatrix(myOutput, outputSize);
    }

    /** 
     * @brief Destructor. 
     */
    ~Flatten() noexcept override = default;

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
     * @brief Get the input gradients of the layer.
     * 
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override { return myInputGradients; }

    /**
     * @brief Get the flattened output of the layer.
     * 
     * @return Matrix holding the output of the layer.
     */
    const Matrix1d& output() const noexcept override { return myOutput; }

    /**
     * @brief Flatten the input from 2D to 1D.
     * 
     * @param[in] input Matrix holding input data.
     * 
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override
    {
        // Return true if the input matches the expected (unpadded) output size.
        constexpr const char* opName{"feedforward in flatten layer"};
        return matchDimensions(myInputGradients.size(), input.size(), opName)
            && isMatrixSquare(input, opName);
    }

     /**
     * @brief Unflatten the output gradients from 1D to 2D.
     * 
     * @param[in] outputGradients Matrix holding output gradients.
     * 
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix1d& outputGradients) noexcept override
    {
         // Return true if the output dimensions match.
        constexpr const char* opName{"backpropagation in flatten layer"};
        return matchDimensions(myOutput.size(), outputGradients.size(), opName);
    }

    Flatten()                          = delete; // No default constructor.
    Flatten(const Flatten&)            = delete; // No copy constructor.
    Flatten(Flatten&&)                 = delete; // No move constructor.
    Flatten& operator=(const Flatten&) = delete; // No copy assignment.
    Flatten& operator=(Flatten&&)      = delete; // No move assignment.

private:
    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Flattened output. */
    Matrix1d myOutput;
};
} // namespace ml::flatten_layer::stub
