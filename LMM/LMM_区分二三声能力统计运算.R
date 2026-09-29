# ============================================================================
# 第三部分：逐个声学维度拟合 Tone * Group 模型，并将 summary 固定效应
# 分别保存到 Excel 的不同工作表。
#
# 参考水平：Tone2、learner
# 每个工作表包含四个固定效应：
#   (Intercept)
#   ToneTone3
#   Groupnative
#   ToneTone3:Groupnative
# ============================================================================

setwd("C:/Users/吴佳琪/Desktop/统计计算/LLM")

library(readxl)
library(lme4)
library(lmerTest)  # 为 summary() 提供 Satterthwaite df、t 和 p
library(openxlsx)

data <- read_xlsx("删除错误数据后的最终表格.xlsx")

# 兼容原始数据中可能存在的小写 location 列名。
# 如果数据中已有 Location，则保持原列不变。
if (!"Location" %in% names(data) && "location" %in% names(data)) {
  data$Location <- data$location
}

# -----------------------------------------------------------------------------
# 1. 设置参考水平
# -----------------------------------------------------------------------------
# 先运行下面两行确认原始水平名称：
levels(factor(data$Tone))
levels(factor(data$Group))

data$Tone <- relevel(factor(data$Tone), ref = "Tone2")
data$Group <- relevel(factor(data$Group), ref = "learner")

# 检查参考水平是否正确
stopifnot(identical(levels(data$Tone)[1], "Tone2"))
stopifnot(identical(levels(data$Group)[1], "learner"))

# -----------------------------------------------------------------------------
# 2. 逐个运行模型
# -----------------------------------------------------------------------------

# ---------- f0range ----------
m3_f0range <- lmer(
  f0range ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_f0range)

# ---------- f0onset ----------
m3_f0onset <- lmer(
  f0onset ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_f0onset)

# ---------- f025 ----------
m3_f025 <- lmer(
  f025 ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_f025)

# ---------- f050 ----------
m3_f050 <- lmer(
  f050 ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_f050)

# ---------- f075 ----------
m3_f075 <- lmer(
  f075 ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_f075)

# ---------- f0offset ----------
m3_f0offset <- lmer(
  f0offset ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_f0offset)

# ---------- max_pos ----------
m3_max_pos <- lmer(
  max_pos ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_max_pos)

# ---------- min_pos ----------
m3_min_pos <- lmer(
  min_pos ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_min_pos)

# ---------- f0_max ----------
m3_f0_max <- lmer(
  f0_max ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_f0_max)

# ---------- f0_min ----------
m3_f0_min <- lmer(
  f0_min ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_f0_min)

# ---------- mean ----------
m3_mean <- lmer(
  mean ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_mean)

# ---------- slope ----------
m3_slope <- lmer(
  slope ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_slope)

# ---------- curve ----------
m3_curve <- lmer(
  curve ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_curve)

# ---------- BLstart ----------
m3_BLstart <- lmer(
  BLstart ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_BLstart)

# ---------- BLslope1 ----------
m3_BLslope1 <- lmer(
  BLslope1 ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_BLslope1)

# ---------- BLbreakpoint ----------
m3_BLbreakpoint <- lmer(
  BLbreakpoint ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_BLbreakpoint)

# ---------- BLslope2 ----------
m3_BLslope2 <- lmer(
  BLslope2 ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_BLslope2)

# ---------- onglide ----------
m3_onglide <- lmer(
  onglide ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_onglide)

# ---------- offglide ----------
m3_offglide <- lmer(
  offglide ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_offglide)

# ---------- overall ----------
m3_overall <- lmer(
  overall ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_overall)

# ---------- Duration ----------
m3_Duration <- lmer(
  Duration ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_Duration)

# ---------- Location ----------
m3_Location <- lmer(
  Location ~ Tone * Group + (1 + Tone | Speaker),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m3_Location)

# -----------------------------------------------------------------------------
# 3. 将每个模型的 summary 固定效应分别写入 Excel
# -----------------------------------------------------------------------------
# 注意：这里的 m3_模型名必须与上面逐个拟合的对象名称一致。
model_list <- list(
  f0range = m3_f0range,
  f0onset = m3_f0onset,
  f025 = m3_f025,
  f050 = m3_f050,
  f075 = m3_f075,
  f0offset = m3_f0offset,
  max_pos = m3_max_pos,
  min_pos = m3_min_pos,
  f0_max = m3_f0_max,
  f0_min = m3_f0_min,
  mean = m3_mean,
  slope = m3_slope,
  curve = m3_curve,
  BLstart = m3_BLstart,
  BLslope1 = m3_BLslope1,
  BLbreakpoint = m3_BLbreakpoint,
  BLslope2 = m3_BLslope2,
  onglide = m3_onglide,
  offglide = m3_offglide,
  overall = m3_overall,
  Duration = m3_Duration,
  Location = m3_Location
)

# 提取 summary() 中的固定效应表，并将行名保留为 coefficient。
extract_summary <- function(model, variable_name) {
  coefficient_table <- as.data.frame(coef(summary(model)), check.names = FALSE)
  coefficient_table$coefficient <- rownames(coefficient_table)
  rownames(coefficient_table) <- NULL
  
  # 将列名改为论文中更容易识别的名称；保留原始 p 值数值。
  names(coefficient_table)[names(coefficient_table) == "Estimate"] <- "beta"
  names(coefficient_table)[names(coefficient_table) == "Std. Error"] <- "SE"
  names(coefficient_table)[names(coefficient_table) == "t value"] <- "t"
  p_column <- grep("Pr\\(", names(coefficient_table), value = TRUE)
  if (length(p_column) == 1) {
    names(coefficient_table)[names(coefficient_table) == p_column] <- "p"
  }
  
  # 把 coefficient 放到第一列。
  coefficient_table <- coefficient_table[, c(
    "coefficient",
    setdiff(names(coefficient_table), "coefficient")
  )]
  coefficient_table$dimension <- variable_name
  
  # dimension和coefficient已明确写在前两列，因此必须同时从setdiff中排除。
  # 否则coefficient会被再次选入，Excel中就会出现coefficient.1重复列。
  coefficient_table <- coefficient_table[, c(
    "dimension",
    "coefficient",
    setdiff(names(coefficient_table), c("dimension", "coefficient"))
  )]
  coefficient_table
}

summary_tables <- Map(extract_summary, model_list, names(model_list))

# -----------------------------------------------------------------------------
# 4. 整理老师要求的总表
# -----------------------------------------------------------------------------
# 每个维度只保留两行：
#   Tone3        = 学习者组三声 - 二声
#   Tone3:Native =（母语者三声 - 二声）-（学习者三声 - 二声）
result_table <- do.call(rbind, lapply(names(summary_tables), function(variable_name) {
  table <- summary_tables[[variable_name]]
  
  tone3_row <- table[table$coefficient == "ToneTone3", , drop = FALSE]
  interaction_row <- table[
    table$coefficient == "ToneTone3:Groupnative",
    ,
    drop = FALSE
  ]
  
  if (nrow(tone3_row) != 1 || nrow(interaction_row) != 1) {
    stop(
      "模型 ", variable_name,
      " 中没有唯一找到 ToneTone3 或 ToneTone3:Groupnative。请检查参考水平。"
    )
  }
  
  data.frame(
    dimension = c(variable_name, variable_name),
    result = c("Tone3", "Tone3:Native"),
    `Estimate (β)` = c(tone3_row$beta, interaction_row$beta),
    SE = c(tone3_row$SE, interaction_row$SE),
    p = c(tone3_row$p, interaction_row$p),
    df = c(tone3_row$df, interaction_row$df),
    t = c(tone3_row$t, interaction_row$t),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}))
rownames(result_table) <- NULL

# p值按心理学论文惯例显示：小于.001写作“<.001”，其余保留三位小数。
result_table$p <- ifelse(
  result_table$p < .001,
  "<.001",
  sprintf("%.3f", result_table$p)
)

output_file <- "第三部分_二三声区分能力_大表.xlsx"
wb <- createWorkbook()

# 主表放在Excel的第一个sheet。
addWorksheet(wb, "结果总表")
writeData(wb, "结果总表", result_table, startRow = 1, withFilter = TRUE)

# 在主表下方解释两种结果的含义。
note_row <- nrow(result_table) + 3
writeData(
  wb,
  "结果总表",
  x = paste0(
    "注：Tone3 = 学习者组三声 − 二声；",
    "Tone3:Native =（母语者组三声 − 二声）−（学习者三声 − 二声）。",
    "Tone和Group的参考水平分别为Tone2和learner。"
  ),
  startRow = note_row,
  startCol = 1,
  colNames = FALSE
)
mergeCells(wb, "结果总表", cols = 1:7, rows = note_row)

header_style <- createStyle(
  fontName = "Times New Roman",
  fontSize = 11,
  textDecoration = "bold",
  halign = "center",
  border = "bottom"
)
body_style <- createStyle(fontName = "Times New Roman", fontSize = 11)
number_style <- createStyle(numFmt = "0.000")

# 设置结果总表格式：Estimate、SE、df和t显示三位小数，p已按论文格式处理。
addStyle(wb, "结果总表", header_style, rows = 1, cols = 1:7, gridExpand = TRUE)
addStyle(wb, "结果总表", body_style,
         rows = 2:(nrow(result_table) + 1), cols = 1:7, gridExpand = TRUE)
addStyle(wb, "结果总表", number_style,
         rows = 2:(nrow(result_table) + 1), cols = c(3:4, 6:7),
         gridExpand = TRUE, stack = TRUE)
addStyle(wb, "结果总表",
         createStyle(fontName = "Times New Roman", fontSize = 10,
                     textDecoration = "italic", wrapText = TRUE),
         rows = note_row, cols = 1:7, gridExpand = TRUE)
freezePane(wb, "结果总表", firstRow = TRUE)
setColWidths(wb, "结果总表", cols = 1, widths = 18)
setColWidths(wb, "结果总表", cols = 2, widths = 18)
setColWidths(wb, "结果总表", cols = 3:7, widths = 15)

for (variable_name in names(summary_tables)) {
  # Excel 工作表名称不能超过31个字符；本列表中的名称均符合要求。
  addWorksheet(wb, variable_name)
  table <- summary_tables[[variable_name]]
  writeData(wb, variable_name, table, withFilter = TRUE)
  
  addStyle(wb, variable_name, header_style, rows = 1,
           cols = seq_len(ncol(table)), gridExpand = TRUE)
  addStyle(wb, variable_name, body_style,
           rows = 2:(nrow(table) + 1),
           cols = seq_len(ncol(table)), gridExpand = TRUE)
  
  # beta、SE、df、t、p 全部保留三位小数显示。
  numeric_columns <- which(names(table) %in% c("beta", "SE", "df", "t", "p"))
  if (length(numeric_columns) > 0) {
    addStyle(wb, variable_name, number_style,
             rows = 2:(nrow(table) + 1),
             cols = numeric_columns,
             gridExpand = TRUE,
             stack = TRUE)
  }
  
  freezePane(wb, variable_name, firstRow = TRUE)
  setColWidths(wb, variable_name, cols = seq_len(ncol(table)), widths = "auto")
}

# 额外添加一个总表，方便横向查看所有模型的交互项。
interaction_table <- do.call(rbind, lapply(summary_tables, function(x) {
  x[x$coefficient == "ToneTone3:Groupnative", , drop = FALSE]
}))
rownames(interaction_table) <- NULL
addWorksheet(wb, "交互项总表")
writeData(wb, "交互项总表", interaction_table, withFilter = TRUE)
addStyle(wb, "交互项总表", header_style, rows = 1,
         cols = seq_len(ncol(interaction_table)), gridExpand = TRUE)
addStyle(wb, "交互项总表", body_style,
         rows = 2:(nrow(interaction_table) + 1),
         cols = seq_len(ncol(interaction_table)), gridExpand = TRUE)
numeric_columns <- which(names(interaction_table) %in% c("beta", "SE", "df", "t", "p"))
addStyle(wb, "交互项总表", number_style,
         rows = 2:(nrow(interaction_table) + 1),
         cols = numeric_columns, gridExpand = TRUE, stack = TRUE)
freezePane(wb, "交互项总表", firstRow = TRUE)
setColWidths(wb, "交互项总表", cols = seq_len(ncol(interaction_table)), widths = "auto")

saveWorkbook(wb, output_file, overwrite = TRUE)

cat("\n完成。Excel 文件已保存到：\n", file.path(getwd(), output_file), "\n")
cat("第一个sheet‘结果总表’包含每个维度的Tone3与Tone3:Native两行结果。\n")
cat("其余sheet保留各模型的完整summary固定效应，便于核对。\n")

