# =============================================================================
# 【用户手动操作步骤】请按以下顺序执行
# 1. 将工作目录修改为您的数据所在路径（第23行）
# 2. 如果数据文件名称不同，修改第26行中的文件名
# 3. 检查 K_range 范围（第105行），可根据计算速度调整
# 4. 运行整个脚本（Ctrl+A，然后 Ctrl+Enter）
# 5. 结果将保存在工作目录下的 "DTW" 文件夹中
# =============================================================================

# ------------------------------ 1. 加载包 ------------------------------------
# 如果未安装，请先运行 install.packages(c("dtwclust", "tidyverse", "readxl", "ggplot2", "cluster"))
library(dtwclust)   # 时间序列聚类
library(tidyverse)  # 数据处理
library(readxl)     # 读取 Excel
library(ggplot2)    # 绘图
library(cluster)    # 轮廓系数

# ------------------------------ 2. 设置工作目录 --------------------------------
# ★★★ 请修改为您的实际路径 ★★★
setwd("C:/Users/吴佳琪/Desktop/统计计算/DTW")   # <--- 手动修改此处

# ------------------------------ 3. 读取数据 ------------------------------------
df <- read_xlsx("运算DTW.xlsx")   # 如果文件名不同，请修改

# 检查数据是否读入成功
if (!exists("df")) stop("数据读取失败，请检查文件路径和名称！")

# ------------------------------ 4. 数据预处理 ------------------------------------
# 提取F0列名
f0_cols <- paste0("F0_", 1:20)

# 定义四个子集（按 Tone 和 Group 筛选）
subsets <- list(
  native_T2  = df %>% filter(Tone == "Tone2" & Group == "native"),
  learner_T2 = df %>% filter(Tone == "Tone2" & Group == "learner"),
  native_T3  = df %>% filter(Tone == "Tone3" & Group == "native"),
  learner_T3 = df %>% filter(Tone == "Tone3" & Group == "learner")
)

# 预处理函数：提取F0矩阵 → 行内Z-score标准化 → 转换为列表
preprocess_for_dtw <- function(data) {
  X <- as.matrix(data[, f0_cols])
  X_scaled <- t(scale(t(X)))        # 行标准化（Z-score）
  series_list <- split(X_scaled, row(X_scaled))
  names(series_list) <- NULL
  return(series_list)
}


# 对所有子集进行预处理
series_list <- lapply(subsets, preprocess_for_dtw)


# ------------------------------ 5. 定义DTW聚类函数 --------------------------------
# 功能：自动确定最佳K（轮廓系数）、执行聚类、保存图表和带标签的数据
run_dtw_subset <- function(subset_name, K_range = 2:6) {
  
  cat("\n========== Processing", subset_name, "==========\n")
  
  # 提取该子集的时间序列数据
  series <- series_list[[subset_name]]
  if (is.null(series)) stop(paste("子集", subset_name, "不存在！"))
  
  # 计算不同K值的平均轮廓系数
  avg_sil <- sapply(K_range, function(k) {
    cl <- tsclust(series, type = "partitional", k = k,
                  distance = "dtw_basic", centroid = "pam",
                  seed = 123, trace = FALSE)
    sil <- silhouette(cl@cluster, dist = cl@distmat)
    mean(sil[, 3])
  })
  
  # 选择最佳K（轮廓系数最大）
  best_k <- K_range[which.max(avg_sil)]
  cat("Best K for", subset_name, ":", best_k, "\n")
  
  # 执行最终聚类
  final_cl <- tsclust(series, type = "partitional", k = best_k,
                      distance = "dtw_basic", centroid = "pam",
                      seed = 123, trace = FALSE)
  
  # 创建输出文件夹
  dtw_folder <- "DTW"
  if (!dir.exists(dtw_folder)) dir.create(dtw_folder)
  
  # 保存序列图（每个簇的所有序列 + 质心）
  png(file.path(dtw_folder, paste0("dtw_series_", subset_name, ".png")),
      width = 1000, height = 600)
  plot(final_cl, type = "series",
       main = paste("DTW Series -", subset_name, "(K=", best_k, ")"))
  dev.off()
  
  # 保存质心图（仅质心）
  png(file.path(dtw_folder, paste0("dtw_centroids_", subset_name, ".png")),
      width = 800, height = 600)
  plot(final_cl, type = "centroids",
       main = paste("DTW Centroids -", subset_name, "(K=", best_k, ")"))
  dev.off()
  
  # 提取聚类标签并保存数据（重置索引避免错位）
  data_sub <- as.data.frame(subsets[[subset_name]])
  rownames(data_sub) <- NULL                  # <--- 加上这一行，强制重置行索引
  data_sub$dtw_cluster <- final_cl@cluster
  readr::write_excel_csv(data_sub,
                         file.path(dtw_folder, paste0(subset_name, "_with_dtw_cluster.csv")))
  
  # 打印簇大小
  print(table(data_sub$dtw_cluster))
  
  # 返回聚类对象
  return(final_cl)
}

# ------------------------------ 6. 运行所有子集 ------------------------------------
# ★★★ 如果计算较慢，可将 K_range 缩小为 2:5 ★★★
all_names <- c("native_T2", "learner_T2", "native_T3", "learner_T3")
results_dtw <- list()
for (name in all_names) {
  results_dtw[[name]] <- run_dtw_subset(name, K_range = 2:6)
}

# ------------------------------ 7. 完成提示 ------------------------------------
cat("\n========== 全部完成！==========\n")
cat("结果保存在工作目录下的 DTW 文件夹中。\n")
cat("包含每个子集的序列图、质心图和带聚类标签的CSV数据。\n")

# 检查 learner_T2 的原始行数与聚类结果数是否完全一致
nrow(subsets$learner_T2)
length(results_dtw$learner_T2@cluster)


