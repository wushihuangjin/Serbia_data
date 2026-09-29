# 加载所需包

library(tidyverse)   # 数据处理
library(factoextra)  # 可视化聚类结果（可选）
library(NbClust)     # 自动推荐最佳 K（可选）

# 1. 数据读入（假设您的数据文件名为 "duration_with_Location.xlsx"）
# 使用 readxl 包读取
library(readxl)
setwd("C:/Users/吴佳琪/Desktop/统计计算/欧式距离聚类")
df <- read_xlsx("删除错误数据后的最终表格.xlsx")
# 2. 定义四个子集
subsets <- list(
  native_T2 = df %>% filter(Tone == "Tone2" & Group == "native"),
  learner_T2 = df %>% filter(Tone == "Tone2" & Group == "learner"),
  native_T3 = df %>% filter(Tone == "Tone3" & Group == "native"),
  learner_T3 = df %>% filter(Tone == "Tone3" & Group == "learner")
)

# 3. 定义聚类函数
# 轮廓系数 --------------------------------------------------------------------

cluster_and_plot <- function(data, subset_name, K_range = 2:10) {
  
  if (!dir.exists("output_plots")) dir.create("output_plots")
  
  f0_cols <- paste0("F0_", 1:20)
  X <- data[, f0_cols]
  X_scaled <- t(scale(t(X)))
  
  # ---------- 1. 肘部图（保留，但不作为最终依据） ----------
  wss <- sapply(K_range, function(k) kmeans(X_scaled, centers = k, nstart = 25)$tot.withinss)
  elbow_plot <- data.frame(K = K_range, WSS = wss) %>%
    ggplot(aes(x = K, y = WSS)) + geom_line() + geom_point() +
    ggtitle(paste("Elbow Method -", subset_name))
  ggsave(paste0("output_plots/elbow_", subset_name, ".png"), elbow_plot, width = 8, height = 6, dpi = 300)
  print(elbow_plot)
  
  # ---------- 2. 轮廓系数图（更客观） ----------
  library(cluster)
  avg_sil <- sapply(K_range, function(k) {
    km <- kmeans(X_scaled, centers = k, nstart = 25)
    sil <- silhouette(km$cluster, dist(X_scaled))
    mean(sil[, 3])
  })
  sil_plot <- data.frame(K = K_range, Avg_Silhouette = avg_sil) %>%
    ggplot(aes(x = K, y = Avg_Silhouette)) + geom_line() + geom_point() +
    ggtitle(paste("Average Silhouette -", subset_name)) +
    ylab("Average Silhouette Width")
  ggsave(paste0("output_plots/silhouette_", subset_name, ".png"), sil_plot, width = 8, height = 6, dpi = 300)
  print(sil_plot)
  
  # ---------- 3. NbClust 综合推荐（可选，可能较慢） ----------
  # library(NbClust)
  # nb <- NbClust(data = X_scaled, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")
  # best_k_nb <- as.numeric(names(which.max(table(nb$Best.nc[1,]))))
  # cat("NbClust recommended K for", subset_name, ":", best_k_nb, "\n")
  
  # 选择最佳 K：基于轮廓系数最高值
  best_k <- K_range[which.max(avg_sil)]
  cat("Best K by silhouette for", subset_name, ":", best_k, "\n")
  
  # 使用最佳 K 进行最终聚类（若需固定 K，可手动修改）
  K_final <- best_k
  set.seed(123)
  km_res <- kmeans(X_scaled, centers = K_final, nstart = 25)
  data$cluster <- as.factor(km_res$cluster)
  
  # ---------- 4. PCA 图 ----------
  pca <- prcomp(X_scaled, scale. = FALSE)
  pca_df <- as.data.frame(pca$x[, 1:2])
  pca_df$cluster <- data$cluster
  pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
    geom_point(alpha = 0.7) +
    ggtitle(paste("PCA -", subset_name, "(K=", K_final, ")"))
  ggsave(paste0("output_plots/pca_", subset_name, ".png"), pca_plot, width = 8, height = 6, dpi = 300)
  print(pca_plot)
  
  # ---------- 5. 平均轮廓图 ----------
  cluster_means <- aggregate(X_scaled, by = list(cluster = data$cluster), FUN = mean)
  mean_curves <- cluster_means %>%
    pivot_longer(cols = starts_with("F0_"), names_to = "Point", values_to = "F0") %>%
    mutate(Point = as.numeric(gsub("F0_", "", Point)))
  curve_plot <- ggplot(mean_curves, aes(x = Point, y = F0, color = cluster, group = cluster)) +
    geom_line(size = 1.2) +
    ggtitle(paste("Mean F0 contours -", subset_name, "(K=", K_final, ")"))
  ggsave(paste0("output_plots/curve_", subset_name, ".png"), curve_plot, width = 8, height = 6, dpi = 300)
  print(curve_plot)
  
  return(list(data = data, kmeans = km_res, pca = pca, best_k = K_final))
}

# 4. 对四个子集分别运行
results <- list()
for (name in names(subsets)) {
  cat("\n========== Processing", name, "==========\n")
  results[[name]] <- cluster_and_plot(subsets[[name]], name, K_range = 2:10)
}

