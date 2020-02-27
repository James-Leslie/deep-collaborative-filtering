# Deep Collaborative Filtering

---
# 1. Rating model
## 1.1. Model architecture
![rating_model](https://github.com/James-Leslie/deep-collaborative-filtering/blob/master/figures/png_images/rating-model.png?raw=true)

### Include user and item bias
![baselines](https://github.com/James-Leslie/deep-collaborative-filtering/blob/master/figures/png_images/baseline.png?raw=true)

# 2. Genre model
## 2.1. Model architecture
Re-use the item embedding layer, but freeze the weights.
![transfer-weights](https://github.com/James-Leslie/deep-collaborative-filtering/blob/master/figures/png_images/transfer-learning.png?raw=true)
