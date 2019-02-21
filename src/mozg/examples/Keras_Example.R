# Clear variables
rm(list=ls())

library(keras)

WORKING.DIR <- "~/git/mozg/src/mozg/examples"
setwd(WORKING.DIR)

mnist <- dataset_mnist()

# Training data/images
train_images <- mnist$train$x
train_labels <- mnist$train$y
# Control images
test_images <- mnist$test$x
test_labels <- mnist$test$y

# 60K images, 28x28 pixel image
str(train_images)
# 60K labels/classifications
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

network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Instead of 2D data for each image 28x28, we pull it into 1D of length 28*28
train_images <- array_reshape(
  train_images,
  c(60000, 28*28)
)
train_images <- train_images / 255

test_images <- array_reshape(
  test_images,
  c(10000, 28*28)
)
test_images <- test_images / 255

# Preparing labels from labels 0-9 to output 0/1 in a positional vector instead
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

network %>% fit(
  train_images,
  train_labels,
  epochs = 5,
  batch_size = 128
)


