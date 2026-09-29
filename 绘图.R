# ================== 加载包 ==================
library(readxl)
library(tidyr)
library(ggplot2)
library(dplyr)
library(stringr)
library(readr)

# ================== 设置路径并读取新数据 ==================
setwd("C:/Users/吴佳琪/Desktop/统计计算/绘图")
data <- read_xlsx("删除错误数据后的最终表格.xlsx")

# 查看列名（确认数据正确）
print(colnames(data))

# ------------- 1. F0 曲线（与之前相同） -------------
f0_cols <- grep("^F0_", colnames(data), value = TRUE)

data_long <- data %>%
  pivot_longer(cols = all_of(f0_cols), names_to = "Point", values_to = "Tvalue") %>%
  mutate(Point = as.numeric(gsub("F0_", "", Point)))

test_avg_f0 <- data_long %>%
  group_by(Group, Point, Tone) %>%
  summarise(
    Tvalue_mean = mean(Tvalue, na.rm = TRUE),
    SD = sd(Tvalue, na.rm = TRUE),
    N = sum(!is.na(Tvalue)),
    SE = SD / sqrt(N),
    .groups = "drop"
  )

p_f0 <- ggplot(test_avg_f0, aes(x = Point, y = Tvalue_mean, color = Tone, group = Tone)) +
  geom_line() +
  geom_point(size = 0.5) +
  geom_errorbar(aes(ymin = Tvalue_mean - SE, ymax = Tvalue_mean + SE), width = 0.3, linewidth = 0.7) +
  facet_grid(. ~ Group) +
  labs(title = "平均 F0 曲线", x = "时间点", y = "平均 F0 (Hz)", color = "声调") +
  theme_bw() +
  theme(strip.background = element_rect(fill = "lightgray"), legend.position = "bottom")
print(p_f0)
ggsave("平均F0_Group_Tone.png", plot = p_f0, width = 10, height = 6, dpi = 300)

# ------------- 2. 其他维度（按实际列名选择） -------------
dimension_cols <- c(
  "f0range", "f0onset", "f025", "f050", "f075", "f0offset", "max_pos", 
  "min_pos", "f0_max", "f0_min", "mean", "slope", "curve", "BLslope1", 
  "BLslope2", "onglide", "offglide", "overall", "Duration"
)

# 检查缺失
missing <- setdiff(dimension_cols, colnames(data))
if (length(missing) > 0) {
  warning("以下列不存在，请检查：", paste(missing, collapse = ", "))
}

data_long_dim <- data %>%
  pivot_longer(cols = all_of(dimension_cols), names_to = "dimension", values_to = "value")

# 汇总并过滤掉无关维度（如 spk_min, spk_max, BLbreakpoint）
test_avg_dim <- data_long_dim %>%
  group_by(Group, dimension, Tone) %>%
  summarise(
    value_mean = mean(value, na.rm = TRUE),
    SD = sd(value, na.rm = TRUE),
    N = sum(!is.na(value)),
    SE = SD / sqrt(N),
    .groups = "drop"
  ) %>%
  filter(!dimension %in% c("spk_min", "spk_max", "BLbreakpoint"))

# ---------- 调整维度标签，使其与原图一致 ----------
test_avg_dim <- test_avg_dim %>%
  mutate(dimension = case_when(
    dimension == "min_pos" ~ "min_pos(turning point)",
    TRUE ~ dimension
  ))

# 按原图分组
test_avg1 <- test_avg_dim %>% filter(dimension %in% c("max_pos", "min_pos(turning point)"))
test_avg2 <- test_avg_dim %>% filter(dimension %in% c("BLslope1", "BLslope2", "onglide", "offglide", "overall", "slope", "curve", "Duration"))
test_avg3 <- test_avg_dim %>% filter(dimension %in% c("f0range", "f0onset", "f025", "f050", "f075", "f0offset", "mean", "f0_max", "f0_min"))

# ------------- 图1：maxmin location -------------
p1 <- ggplot(test_avg1, aes(x = Group, y = value_mean, color = Tone, group = Tone)) +
  geom_point(size = 0.5) +
  geom_errorbar(aes(ymin = value_mean - SE, ymax = value_mean + SE), width = 0.3, linewidth = 0.7) +
  facet_grid(. ~ dimension) +   # 标签已经过 mutate 处理，显示为 "min_pos(turning point)"
  labs(title = "max/min location", x = "Group", y = "index") +
  theme_bw() +
  theme(strip.background = element_rect(fill = "lightgray"), legend.position = "bottom")
print(p1)
ggsave("maxmin_location.png", plot = p1, width = 8, height = 5, dpi = 300)

# ------------- 图2：contour & slope index -------------
p2 <- ggplot(test_avg2, aes(x = Group, y = value_mean, color = Tone, group = Tone)) +
  geom_point(size = 0.5) +
  geom_errorbar(aes(ymin = value_mean - SE, ymax = value_mean + SE), width = 0.3, linewidth = 0.7) +
  facet_grid(. ~ dimension) +
  labs(title = "contour & slope index", x = "Group", y = "index") +
  theme_bw() +
  theme(strip.background = element_rect(fill = "lightgray"), legend.position = "bottom")
print(p2)
ggsave("dimension_contour.png", plot = p2, width = 10, height = 6, dpi = 300)

# ------------- 图3：Tvalue height index -------------
p3 <- ggplot(test_avg3, aes(x = Group, y = value_mean, color = Tone, group = Tone)) +
  geom_point(size = 0.5) +
  geom_errorbar(aes(ymin = value_mean - SE, ymax = value_mean + SE), width = 0.3, linewidth = 0.7) +
  facet_grid(. ~ dimension) +
  labs(title = "Tvalue height index", x = "Group", y = "index") +
  theme_bw() +
  theme(strip.background = element_rect(fill = "lightgray"), legend.position = "bottom")
print(p3)
ggsave("Tvalue_height_index.png", plot = p3, width = 12, height = 6, dpi = 300)

