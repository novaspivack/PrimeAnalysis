Advanced Machine Learning and Analytics for Prime Gap Distribution Investigation

This code conducts a comprehensive investigation into the properties and patterns of prime gaps, leveraging a combination of data analysis, feature engineering, machine learning, and advanced mathematical techniques. The primary objective is to discern underlying structures and predictive indicators within the seemingly random distribution of prime gaps.

NOTE: This code was implemented using Multiprocessing on MacOS. 

The initial phase involves generating a dataset of the first N primes and their corresponding gaps. Beyond simply computing the gaps, the code performs extensive feature engineering, deriving a rich set of attributes for each gap.

These features encompass:

Factor Analysis: For the composite numbers residing within each gap, the code analyzes their prime factorization. 

Features include the total number of prime factors, the number of unique prime factors, the maximum and minimum prime factors, the mean and standard deviation of the factors, the density of factors (total factors divided by the gap size), and the Shannon entropy of the factor distribution. 

Additionally, the square root of each factor is used to derive further features, such as the mean, sum, and standard deviation of the square roots.

Prime Type Identification: The code classifies the primes flanking each gap according to their type (e.g., twin, Sophie Germain, safe, Mersenne). This classification is implemented using optimized algorithms and caching for efficiency.

Gap Properties: The gap size itself is included as a feature, along with its residues modulo 6 and 30. These modulo residues are chosen due to their relevance in prime gap patterns.
Advanced Features: The code constructs additional features based on combinations and transformations of the initial features. This includes rolling statistics (mean, standard deviation, skewness, kurtosis) of the gap size, power and logarithmic transformations of factor-related features, interaction terms (products, ratios, sums) between key features, lag features of gap size and factor density, difference features of gap size, and frequency domain features derived from the Fast Fourier Transform (FFT) of the gap sequence. Furthermore, pattern-based features like distance to the nearest peak in the gap sequence and indicators for local maxima and minima are included. Composite features, combining multiple aspects of factor analysis, gap dynamics, and frequency characteristics, are also engineered to capture more complex relationships.

Following feature engineering, the code performs initial clustering using K-Means with an optimized number of clusters determined by the silhouette score. This initial clustering serves as a foundation for subsequent analyses.

The core of the analysis lies in training predictive models to forecast gap sizes. The code employs a diverse set of models, including:

Random Forest Regressor: This ensemble method is used for its robustness and ability to capture nonlinear relationships.

XGBoost Regressor: Another gradient boosting algorithm known for its high performance and efficiency.

Linear Regression: A simple linear model serves as a baseline for comparison.

Neural Network: A feedforward neural network with multiple layers and regularization is implemented using TensorFlow/Keras. The architecture includes batch normalization, dropout, and a robust loss function (Huber loss) to enhance stability and prevent overfitting. Training utilizes callbacks for early stopping, learning rate reduction, and model checkpointing.

Cluster-Specific Models: The code trains separate models to predict cluster membership, gap size given cluster membership, and the next cluster in the sequence. These models aim to capture the specific dynamics within and between clusters.

Ensemble and Stacking Models: To further improve predictive performance, the code constructs ensemble and stacking models. The ensemble model combines predictions from multiple base models using a weighted average. The stacking model uses base model predictions as input to a meta-learner (typically linear regression).

Model evaluation is performed using time series cross-validation to prevent data leakage. Metrics include mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), R-squared, and normalized versions of MSE and RMSE. Prediction intervals are computed using bootstrapping for applicable models.

Beyond predictive modeling, the code performs several advanced analyses:

Chaos Metrics: Lyapunov exponents and divergence rates are calculated to quantify the sensitivity to initial conditions in the gap sequence.

Superposition Patterns: The code investigates the presence of multimodality and entropy in the distribution of features, indicative of superposition-like phenomena.

Wavelet Analysis: Wavelet decomposition is used to analyze the gap sequence at different scales, revealing potential hierarchical structures.

Fractal Dimension Analysis: The box-counting method is employed to estimate the fractal dimension of the gap sequence, providing a measure of its geometric complexity.

Phase Space Analysis: The code reconstructs the phase space of the gap sequence using time-delayed embedding and analyzes properties like embedding dimension and recurrence.

Recurrence Plot Analysis: Recurrence plots are generated to visualize recurring patterns and temporal dependencies in the gap sequence.

Feature importance is assessed using various methods, including model-based importance scores (Random Forest, XGBoost), correlation with the target variable, and SHAP values.

Feature selection is performed based on a combination of these importance scores and cross-validated model performance.

Feature interaction analysis includes pairwise correlations, mutual information, nonlinear relationship detection, and SHAP interaction values.

Feature stability is evaluated using bootstrapping and analysis of temporal and value range stability. Network analysis of feature interactions is conducted using graph theory metrics like degree centrality, betweenness centrality, and clustering coefficient.

Finally, the code generates a detailed report summarizing all findings, including statistical descriptions, pattern analysis results, model performance metrics, feature importance rankings, and visualizations. The report also includes an executive summary highlighting the key discoveries and their implications. Interactive visualizations are generated using Plotly to facilitate exploration of the data and results. Throughout the code, robust error handling and numerical stability measures are implemented to ensure reliable execution and prevent issues with large datasets and complex computations.
