# Feature-Extraction-using-PCA-and-Autoencoders
1. Objective of the Assignment

The goal of this assignment is to study and compare different feature extraction techniques, focusing primarily on Principal Component Analysis (PCA) and Autoencoders, and to understand how these learned representations affect downstream tasks such as classification and reconstruction. The assignment explores both classical linear methods and modern neural network–based approaches, highlighting their strengths, limitations, and practical behavior on real image datasets such as CIFAR-10 and MNIST.

2. Dataset Preparation and Preprocessing

The CIFAR-10 dataset is used as the primary benchmark for feature extraction. Since PCA and linear autoencoders operate more naturally on vectorized inputs, the RGB images are first converted to grayscale to reduce dimensionality while retaining meaningful visual structure. All images are normalized and split into training and test sets using a stratified 70/30 split to preserve class balance. Flattening is applied only where required, while spatial structure is preserved for convolutional models. This preprocessing ensures fair and stable comparisons across all methods.

3. PCA and Randomized PCA for Feature Extraction

In the first task, PCA is applied to the centered grayscale CIFAR-10 images, retaining enough principal components to capture 95% of the total variance. This automatically determines the effective latent dimensionality rather than fixing it arbitrarily. Randomized PCA is then applied using the same number of components to evaluate whether faster approximate methods compromise feature quality. Logistic regression is trained on both PCA and Randomized PCA features using a one-vs-rest strategy for multi-class classification. ROC curves and AUC scores are computed for each class, providing a detailed and class-wise evaluation rather than relying solely on accuracy.

4. PCA vs Randomized PCA Analysis

The results show that both PCA and Randomized PCA select nearly identical numbers of components for 95% variance retention. The mean AUC values across all classes are also extremely close, indicating that Randomized PCA produces features of comparable quality for this task. Visual inspection of the ROC curves further confirms this, as the curves for PCA and Randomized PCA almost overlap for most classes. This suggests that Randomized PCA is a reliable and computationally efficient alternative to standard PCA for large-scale problems.

5. Interpretation of PCA Eigenvectors

The leading PCA components, when reshaped back into image form, reveal meaningful visual patterns such as edges, gradients, and low-frequency structures. These eigenvectors capture the most dominant variations in the dataset and resemble classical image filters. This visualization helps build intuition about how PCA compresses data by preserving global structure rather than fine details.

6. Linear Tied-Weight Autoencoder

To compare PCA with learned linear representations, a tied-weight linear autoencoder is implemented using TensorFlow. The encoder and decoder share weights, making the architecture closely related to PCA while still being learned via gradient descent. The model is trained to minimize reconstruction error, with an additional normalization step applied to stabilize training. The learned weight vectors are visualized and compared with PCA eigenvectors, revealing similar edge-like and structured patterns.

7. PCA vs Linear Autoencoder Comparison

While both PCA and the linear autoencoder capture dominant data directions, there are key conceptual differences. PCA enforces orthogonality and maximizes variance, whereas the autoencoder minimizes reconstruction error without orthogonality constraints. As a result, the autoencoder features appear more localized and less globally structured. Despite these differences, the similarity of visual patterns confirms that linear autoencoders can approximate PCA-like behavior under appropriate constraints.

8. Convolutional and Feedforward Autoencoders

In the next task, more expressive autoencoder architectures are explored. A convolutional autoencoder is trained directly on image data, preserving spatial locality through convolution and pooling operations. This is compared against single-layer and three-layer fully connected autoencoders trained on flattened inputs. Reconstruction error on the test set is used as the evaluation metric for all models.

9. Reconstruction Performance Analysis

Among the tested architectures, the convolutional autoencoder achieves the lowest reconstruction error, confirming that convolutional layers are better suited for image data. The single hidden-layer feedforward autoencoder performs reasonably well, while the deeper three-layer model performs significantly worse. This highlights an important lesson: increasing depth does not automatically improve performance, especially when activation functions and layer sizes are not carefully tuned.

10. MNIST Autoencoder and 7-Segment Classification

To demonstrate feature reuse, a convolutional autoencoder is trained on MNIST digits, and the learned latent features are extracted. These features are then used as inputs to a small multilayer perceptron that predicts a 7-segment display representation of digits. This task intentionally simplifies the output space, making classification errors easier to analyze and interpret.

11. Confusion Matrix Analysis

The confusion matrix shows strong performance overall, with most predictions concentrated along the diagonal. Certain digits such as ‘1’ are classified very reliably, while others such as ‘4’, ‘5’, ‘8’, and ‘9’ show more confusion. These errors can be explained by similarities in their 7-segment representations, where multiple digits differ by only one segment. This demonstrates how representation choice directly affects downstream classification behavior.

12. Final Observations and Key Takeaways

This assignment clearly illustrates the trade-offs between classical and neural feature extraction techniques. PCA provides interpretable, stable, and efficient features, while autoencoders offer flexibility and improved performance when spatial structure is preserved. Randomized PCA proves to be a practical alternative with minimal loss in performance. The experiments also reinforce that architectural choices matter more than raw depth and that feature quality directly impacts classification outcomes.

13. Conclusion

Overall, this project provides a comprehensive comparison of PCA, Randomized PCA, and various autoencoder architectures for feature extraction. By combining quantitative metrics, visual analysis, and downstream evaluation, it offers a well-rounded understanding of representation learning. The results emphasize that the best method depends on the data structure and task requirements, and that careful design often matters more than model complexity.
