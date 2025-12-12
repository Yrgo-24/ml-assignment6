/**
 * @brief Machine learning factory stub.
 */
#pragma once

#include <memory>

#include "ml/act_func/none.h"
#include "ml/conv_layer/stub/conv.h"
#include "ml/conv_layer/stub/max_pool.h"
#include "ml/dense_layer/stub/dense.h"
#include "ml/factory/interface.h"
#include "ml/flatten_layer/stub/flatten.h"

namespace ml::factory::stub
{
/**
 * @brief Machine learning factory stub.
 * 
 *        This class is non-copyable and non-movable.
 */
class Factory final : public Interface
{
public:
    /** 
     * @brief Constructor. 
     */
    Factory() noexcept = default;

    /**
     * @brief Destructor.
     */
    ~Factory() noexcept override = default;

    /**
     * @brief Create an activation function.
     * 
     * @param[in] type The type of activation function to create.
     * 
     * @return Pointer to the new activation function.
     */
    ActFuncPtr actFunc(const act_func::Type type) override
    {
        (void) (type);
        return std::make_unique<act_func::None>();
    }

    /**
     * @brief Create a convolutional layer.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     * @param[in] actFunc Activation function to use.
     * 
     * @return Pointer to the new convolutional layer.
     */
    ConvLayerPtr convLayer(const std::size_t inputSize, const std::size_t kernelSize, 
                           const act_func::Type actFunc) override
    {
        return std::make_unique<conv_layer::stub::Conv>(inputSize, kernelSize, actFunc);
    }

    /**
     * @brief Create a dense layer.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Output size. Must be greater than 0.
     * @param[in] actFunc Activation function to use.
     * 
     * @return Pointer to the new dense layer.
     */
    DenseLayerPtr denseLayer(const std::size_t inputSize, const std::size_t outputSize, 
                             const act_func::Type actFunc) override
    {
        return std::make_unique<dense_layer::stub::Dense>(inputSize, outputSize, actFunc);
    }

    /**
     * @brief Create a flatten layer.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     */
    FlattenLayerPtr flattenLayer(const std::size_t inputSize) override
    {
        return std::make_unique<flatten_layer::stub::Flatten>(inputSize);
    }

    /**
     * @brief Create a max pooling layer.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] poolSize Pool size. Must divide the input size.
     * 
     * @return Pointer to the new max pooling layer.
     */
    ConvLayerPtr maxPoolLayer(const std::size_t inputSize, const std::size_t poolSize) override
    {
        return std::make_unique<conv_layer::stub::MaxPool>(inputSize, poolSize);
    }

    Factory(const Factory&)            = delete; // No copy constructor.
    Factory(Factory&&)                 = delete; // No move constructor.
    Factory& operator=(const Factory&) = delete; // No copy assignment.
    Factory& operator=(Factory&&)      = delete; // No move assignment.
};
} // namespace ml::factory::stub