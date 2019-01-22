library(keras)

mnist <- dataset_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

str(train_images)
str(train_labels)

network <- keras_model_sequential() %>%
  layer_dense(
    units = 512,
    activation = "relu",
    input_shape = c(28 * 28)
  ) %>%
  layer_dense(
    units = 10,
    activation = "softmax"
  )
