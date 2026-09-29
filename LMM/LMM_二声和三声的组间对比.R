# ============================================================
# 脚本功能：自动建模 + 事后检验，并将 p 值智能格式化为
#          "<0.001"（若小于0.001）或保留三位小数，导出 Excel。
# ============================================================

library(readxl); library(lme4); library(lmerTest)
library(emmeans); library(dplyr); library(tidyr); library(openxlsx)

# 读取数据
setwd("C:/Users/吴佳琪/Desktop/统计计算/LLM")
data <- read_xlsx("删除错误数据后的最终表格.xlsx")
data$Tone    <- as.factor(data$Tone)
data$Group   <- as.factor(data$Group)
data$Speaker <- as.factor(data$Speaker)

# 因变量列表（注意大小写，按实际数据调整）
dim_list <- c("f0range", "f0onset", "f025", "f050", "f075", "f0offset",
              "max_pos", "min_pos", "f0_max", "f0_min", "mean", "slope",
              "curve", "BLstart", "BLslope1", "BLbreakpoint", "BLslope2",
              "onglide", "offglide", "overall", "Duration", "Location")
dim_list <- intersect(dim_list, colnames(data))

emm_results <- list()

for (dv in dim_list) {
  
  cat("\n========== 正在处理因变量：", dv, "==========\n")
  
  formula <- as.formula(paste0(dv, " ~ Tone * Group + (1 + Tone | Speaker)"))
  model <- lmer(formula, data = data,
                control = lmerControl(optimizer = "bobyqa",
                                      optCtrl = list(maxfun = 2e5)))
  
  # 事后检验
  emm <- emmeans(model, pairwise ~ Group | Tone)
  contrast_result <- as.data.frame(emm$contrasts)
  contrast_result$Dimension <- dv
  
  # ----- 数值格式化（保留三位小数）-----
  # 1. 先对 estimate, SE, df, t.ratio 这些数值保留3位小数
  num_cols <- c("estimate", "SE", "df", "t.ratio")  # 只选择这些列
  num_cols <- intersect(num_cols, colnames(contrast_result))
  contrast_result[num_cols] <- lapply(contrast_result[num_cols], function(x) round(x, 3))
  
  # 2. 特别处理 p.value 列
  if ("p.value" %in% colnames(contrast_result)) {
    # 如果原始 p 值 < 0.001，显示为 "<0.001"；否则保留三位小数
    contrast_result$p.value <- ifelse(
      contrast_result$p.value < 0.001,
      "<0.001",
      as.character(round(contrast_result$p.value, 3))
    )
  }
  # ------------------------------------
  
  emm_results[[dv]] <- contrast_result
  print(head(contrast_result))  # 预览
}

# 写入 Excel
wb <- createWorkbook()
for (dv in names(emm_results)) {
  sheet_name <- substr(dv, 1, 31)
  addWorksheet(wb, sheetName = sheet_name)
  writeData(wb, sheet = sheet_name, x = emm_results[[dv]])
}
saveWorkbook(wb, file = "PostHoc_Group_by_Tone_Formatted.xlsx", overwrite = TRUE)
cat("\n结果已保存至：PostHoc_Group_by_Tone_Formatted.xlsx\n")
