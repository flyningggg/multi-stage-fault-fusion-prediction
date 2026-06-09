# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 800)
        MainWindow.setWindowTitle("油气区断裂网络连通性智能分析与预测系统 v1.0")
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # 主布局 (垂直)
        self.main_vbox = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_vbox.setContentsMargins(10, 10, 10, 10)

        # ==========================================
        # 上部：控制按键区 (你原来的三排排版)
        # ==========================================
        self.top_frame = QtWidgets.QFrame(self.centralwidget)
        self.top_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.top_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 5px; }")
        self.top_layout = QtWidgets.QVBoxLayout(self.top_frame)
        self.top_layout.setSpacing(4)
        self.top_layout.setContentsMargins(6, 6, 6, 6)

        # --- 第零排：数据源切换 ---
        self.row0_layout = QtWidgets.QHBoxLayout()
        self.lbl_data_source = QtWidgets.QLabel("数据源：")
        self.lbl_data_source.setToolTip("当前工区：加载 program/MY 下迹线与研究区 GeoJSON，作为以下所有地图与拓扑分析的底数。")
        self.combo_data_source = QtWidgets.QComboBox()
        self.combo_data_source.addItems([
            "塔里木盆地英买2 (MY)",
        ])
        self.combo_data_source.setMinimumWidth(160)
        self.combo_data_source.setMinimumHeight(35)
        self.combo_data_source.setToolTip("当前仅配置英买 2 区：迹线/研究区为 MY 下 GeoJSON，融合分析使用对应网格 CSV")

        # 通用下拉框样式（确保下拉列表背景白色、文字深色，避免系统主题导致白字白底看不见）
        _combo_style = """
            QComboBox {
                background-color: #ffffff;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 13px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #2c3e50;
                selection-background-color: #3498db;
                selection-color: #ffffff;
                border: 1px solid #bdc3c7;
                outline: none;
            }
        """
        # 仅数据源下拉略放大字号（框内与弹出列表），其余下拉仍用 _combo_style
        _combo_style_data_source = _combo_style.replace(
            "font-size: 13px;",
            "font-size: 15px;",
        ).replace(
            "QComboBox QAbstractItemView {\n                background-color:",
            "QComboBox QAbstractItemView {\n                font-size: 15px;\n                background-color:",
        )
        self.combo_data_source.setStyleSheet(_combo_style_data_source)
        self.row0_layout.addWidget(self.lbl_data_source)
        self.row0_layout.addWidget(self.combo_data_source)
        # 将原第三排下方“打开目录/取消任务”按钮挪到顶部，替代提示位
        self.btn_open_model_dir = QtWidgets.QPushButton("打开 model/")
        self.btn_open_model_dir.setToolTip("打开当前程序目录下的 model 文件夹（含 xgboost_reg.json、预测导出等）。")
        self.btn_open_processed_dir = QtWidgets.QPushButton("打开 data/processed/")
        self.btn_open_processed_dir.setToolTip("打开导出的聚类/流水线中间结果 CSV 等所在目录。")
        self.btn_cancel_task = QtWidgets.QPushButton("取消任务")
        self.btn_cancel_task.setToolTip("取消正在后台执行的长任务（如一键空间-拓扑融合）。")
        _row0_btn_style = "QPushButton { font-size: 12px; padding: 2px 6px; }"
        self.row0_layout.setSpacing(4)
        for b in (self.btn_open_model_dir, self.btn_open_processed_dir, self.btn_cancel_task):
            b.setMinimumHeight(28)
            b.setMaximumHeight(30)
            b.setStyleSheet(_row0_btn_style)
            self.row0_layout.addWidget(b)
        self.row0_layout.addStretch()

        # 右上参数面板：将流程提示 + 参数编辑集中到右上角
        self.top_right_panel = QtWidgets.QWidget()
        self.top_right_panel.setStyleSheet(
            "background:#f3f6fb; border:1px solid #d4dbe6; border-radius:4px;"
        )
        self.top_right_layout = QtWidgets.QVBoxLayout(self.top_right_panel)
        self.top_right_layout.setContentsMargins(8, 4, 8, 4)
        self.top_right_layout.setSpacing(0)

        # --- 运行参数（GUI 覆盖 config.yaml）---
        self.row_ml_params_layout = QtWidgets.QHBoxLayout()
        self.row_ml_params_layout.setContentsMargins(0, 0, 0, 0)
        self.row_ml_params_layout.setSpacing(6)
        self.lbl_kmeans_k = QtWidgets.QLabel("聚类 k:")
        self.lbl_kmeans_k.setToolTip("属性融合中 KMeans 簇数，写入本次运行；仍可在 config.yaml 中保留默认值。")
        self.spin_kmeans_k = QtWidgets.QSpinBox()
        self.spin_kmeans_k.setRange(2, 24)
        self.spin_kmeans_k.setValue(4)
        self.lbl_train_target = QtWidgets.QLabel("训练目标列:")
        self.lbl_train_target.setToolTip("训练 XGBoost 时使用的目标列；下拉为自动推荐的数值列，也可手动编辑。")
        self.combo_train_target = QtWidgets.QComboBox()
        self.combo_train_target.setEditable(True)
        self.combo_train_target.setMinimumWidth(220)
        self.lbl_grid_step = QtWidgets.QLabel("网格步长(m):")
        self.lbl_grid_step.setToolTip("等值线/轮廓图采样时的单元边长；与 export_grid.cell_width 一致思路。")
        self.dspin_grid_step = QtWidgets.QDoubleSpinBox()
        self.dspin_grid_step.setRange(10.0, 50000.0)
        self.dspin_grid_step.setDecimals(1)
        self.dspin_grid_step.setValue(750.0)
        for w in (
            self.lbl_kmeans_k,
            self.spin_kmeans_k,
            self.lbl_train_target,
            self.combo_train_target,
            self.lbl_grid_step,
            self.dspin_grid_step,
        ):
            self.row_ml_params_layout.addWidget(w)
        self.top_right_layout.addLayout(self.row_ml_params_layout)
        self.top_right_panel.setFixedHeight(42)
        self.top_right_panel.setMinimumWidth(640)
        self.row0_layout.addWidget(self.top_right_panel)
        self.top_layout.addLayout(self.row0_layout)

        # --- 第一排：基础地质与拓扑绘图 ---
        self.row1_layout = QtWidgets.QHBoxLayout()
        self.btn_yuantu = QtWidgets.QPushButton("原断裂数据地图")
        self.btn_fenleihou = QtWidgets.QPushButton("分类后数据地图")
        self.btn_relitu = QtWidgets.QPushButton("断裂密度热力图")
        self.btn_azimuth = QtWidgets.QPushButton("方位角集图")
        self.btn_meiguitu = QtWidgets.QPushButton("方向玫瑰图")
        self.btn_sanyuantu = QtWidgets.QPushButton("各类别三元图")
        self.btn_guanxi = QtWidgets.QPushButton("交叉与相邻关系")
        self.btn_b = QtWidgets.QPushButton("确定分支和节点")
        self.btn_a = QtWidgets.QPushButton("长度分布拟合")

        self.btn_yuantu.setToolTip(
            "【原图】在研究区范围内绘制原始断裂迹线，用于检查几何范围与坐标系是否正确。"
        )
        self.btn_fenleihou.setToolTip(
            "【分类图】按 fractopo 规则对迹线做分支类型（CC/CI/II）等着色，看拓扑分类后的空间分布。"
        )
        self.btn_relitu.setToolTip(
            "【密度】沿迹线采样后做核密度估计，颜色表示断裂空间集中程度（非网格属性）。"
        )
        self.btn_azimuth.setToolTip(
            "【方位集】按设定的方位组（如 N-S / E-W）给迹线着色，看优势走向与分组。"
        )
        self.btn_meiguitu.setToolTip(
            "【玫瑰图】迹线/分支方位角的极坐标玫瑰图，看走向集中度。"
        )
        self.btn_sanyuantu.setToolTip(
            "【三元图】节点类型 X/Y/I 或分支端点组合的比例三角图，看拓扑端元结构。"
        )
        self.btn_guanxi.setToolTip(
            "【关系图】方位集之间的交叉（cross-cut）与相邻（abutting）等邻接关系示意。"
        )
        self.btn_b.setToolTip(
            "【分支与节点】左右两图：迹线+研究区与节点类型、分支类型对比，核实拓扑识别结果。"
        )
        self.btn_a.setToolTip(
            "【长度分布】迹线与分支长度直方图 + 幂律/对数正态等拟合曲线，用于长度标度分析。"
        )

        _sp_expand = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        _row1_style = "QPushButton { font-size: 12px; padding: 2px 4px; }"
        self.row1_layout.setSpacing(4)
        for btn in [self.btn_yuantu, self.btn_fenleihou, self.btn_relitu, self.btn_azimuth,
                    self.btn_meiguitu, self.btn_sanyuantu, self.btn_guanxi, self.btn_b, self.btn_a]:
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)
            btn.setSizePolicy(_sp_expand)
            btn.setStyleSheet(_row1_style)
            self.row1_layout.addWidget(btn)
        self.top_layout.addLayout(self.row1_layout)

        # --- 第二排：视图与参数提取 ---
        self.row2_layout = QtWidgets.QHBoxLayout()
        self.combo_topo = QtWidgets.QComboBox()
        self.combo_topo.addItems(["请选择拓扑视图...", "拓扑化后断裂数据地图1", "拓扑化后断裂数据地图2"])
        self.combo_params = QtWidgets.QComboBox()
        self.combo_params.addItems(["下拉选择绘制参数...", "Fracture Intensity B21 / P21 (断裂强度)",
                                    "Trace Max/Min/Mean Length (迹线长度分布)", "Dimensionless Intensity (无量纲强度)",
                                    "Number of Traces (实际迹线数量)", "Branch Max/Min/Mean Length", "Node Count",
                                    "Branch Count"])
        self.btn_tuopushuxing = QtWidgets.QPushButton("显示拓扑属性数据")

        self.combo_topo.setToolTip(
            "【拓扑化视图】选「1」：迹线+节点（按 X/Y/I 着色）；选「2」：分支（C-C/C-I/I-I 等）+ 节点散点。\n"
            "切换选项后会自动重绘，请稍候。"
        )
        self.combo_params.setToolTip(
            "【等值线/平滑图参数】先在此处选好网格指标，再点同一排触发轮廓图流程（与程序内逻辑绑定）。\n"
            "各项对应断裂强度、长度、无量纲强度、迹线数、分支长度等。"
        )
        self.btn_tuopushuxing.setToolTip(
            "在左侧文本框列出当前网络的拓扑标量参数表（如 B21、P21、连接度等整体指标）。"
        )

        self.combo_topo.setStyleSheet(_combo_style)
        self.combo_params.setStyleSheet(_combo_style)
        self.row2_layout.setSpacing(4)
        for widget in [self.combo_topo, self.combo_params, self.btn_tuopushuxing]:
            widget.setMinimumHeight(32)
            widget.setMaximumHeight(36)
            self.row2_layout.addWidget(widget)
        self.row2_layout.addStretch()
        self.top_layout.addLayout(self.row2_layout)

        # --- 第三排：机器学习与属性融合 ---
        self.row3_layout = QtWidgets.QHBoxLayout()
        self.combo_fusion = QtWidgets.QComboBox()
        self.combo_fusion.addItems(["融合方法: PCA", "融合方法: 自编码器", "融合方法: UMAP", "融合方法: VAE"])
        self.btn_ronghe = QtWidgets.QPushButton("执行属性融合分析")
        self.btn_guoji_weighted = QtWidgets.QPushButton("高价值属性加权融合")
        self.btn_guoji_compare = QtWidgets.QPushButton("融合对比(加权vsGAT)")
        self.btn_k_helper = QtWidgets.QPushButton("选k辅助")
        self.btn_guoji_train = QtWidgets.QPushButton("训练XGBoost模型")
        self.lbl_shap_features = QtWidgets.QLabel("SHAP关注特征:")
        self.combo_shap_features = QtWidgets.QComboBox()
        self.combo_shap_features.setMinimumWidth(220)
        self.combo_shap_features.setToolTip(
            "列出当前数据源网格CSV经特征筛选后、实际进入XGBoost的属性。\n"
            "选「全部」时摘要图按 SHAP 默认顺序；选某一列时该特征在图中排在最前便于查看。\n（需先训练模型再点 SHAP。）"
        )
        self.combo_shap_features.addItem("全部（默认顺序）")
        self.btn_guoji_shap = QtWidgets.QPushButton("SHAP可解释分析")
        self.btn_spatial = QtWidgets.QPushButton("一键空间-拓扑融合")
        self.btn_export_results = QtWidgets.QPushButton("导出结果")

        self.combo_fusion.setToolTip(
            "【融合方法】将多列网格拓扑属性降维到 2 维（PCA / 自编码器 / UMAP / VAE），再做 KMeans 聚类。\n"
            "自编码器与 VAE 需 PyTorch；UMAP 需 umap-learn。"
        )
        self.btn_ronghe.setToolTip(
            "读取当前工区网格 CSV，按左侧所选方法做属性融合 + 聚类；右侧显示潜空间散点与空间网格聚类图。"
        )
        self.btn_guoji_weighted.setToolTip(
            "对高价值连通类属性加权，得到每网格一维「勘探价值」得分；结果在左侧文本框，不弹大图。"
        )
        self.btn_guoji_compare.setToolTip(
            "对比规则加权融合与 GAT 图网络融合得分分布（箱线图）。需安装 PyG；失败时会跳过 GAT 并提示。"
        )
        self.btn_k_helper.setToolTip(
            "基于当前融合特征计算不同 k 的 Inertia / Silhouette / Davies-Bouldin 曲线，辅助选择聚类数。"
        )
        self.btn_guoji_train.setToolTip(
            "用特征工程后的矩阵训练 XGBoost（配置见 config.yaml）。需先有英买 2 等网格 CSV。"
        )
        self.btn_guoji_shap.setToolTip(
            "对已训练模型做 SHAP 特征重要性；摘要图在右侧。需先训练；可与「SHAP关注特征」下拉联用。"
        )
        self.btn_spatial.setToolTip(
            "【流水线】特征工程 → 加权融合 →（可选）GAT → XGBoost → SHAP 一键跑完；耗时较长，结果写 processed。"
        )
        self.btn_export_results.setToolTip(
            "汇总显示最近一次聚类/训练/SHAP/一键流水线的导出文件路径（CSV/GPKG/图片/模型）。"
        )

        self.combo_fusion.setStyleSheet(_combo_style)
        self.combo_shap_features.setStyleSheet(_combo_style)
        _row3_btn_style = "QPushButton { font-size: 12px; padding: 2px 4px; }"
        self.row3_layout.setSpacing(4)
        # 融合下拉宽度限制，避免撑太宽
        self.combo_fusion.setMaximumWidth(150)
        self.combo_shap_features.setMinimumWidth(140)
        self.combo_shap_features.setMaximumWidth(200)
        for widget in [self.combo_fusion, self.btn_ronghe, self.btn_guoji_weighted, self.btn_guoji_compare,
                       self.btn_k_helper,
                       self.btn_guoji_train]:
            widget.setMinimumHeight(32)
            widget.setMaximumHeight(36)
            if isinstance(widget, QtWidgets.QPushButton):
                widget.setSizePolicy(_sp_expand)
                widget.setStyleSheet(_row3_btn_style)
            self.row3_layout.addWidget(widget)
        self.row3_layout.addWidget(self.lbl_shap_features)
        self.row3_layout.addWidget(self.combo_shap_features)
        for widget in [self.btn_guoji_shap, self.btn_spatial, self.btn_export_results]:
            widget.setMinimumHeight(32)
            widget.setMaximumHeight(36)
            widget.setSizePolicy(_sp_expand)
            widget.setStyleSheet(_row3_btn_style)
            self.row3_layout.addWidget(widget)
        self.top_layout.addLayout(self.row3_layout)

        self.main_vbox.addWidget(self.top_frame)

        # ==========================================
        # 下部：左右分栏结构
        # ==========================================
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # --- 左侧：配置摘要 + 信息与日志输出区 ---
        self.left_panel = QtWidgets.QWidget(self.splitter)
        self.left_panel_layout = QtWidgets.QVBoxLayout(self.left_panel)
        self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.left_panel_layout.setSpacing(8)

        self.config_summary_title = QtWidgets.QLabel("当前配置摘要")
        self.config_summary_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2c3e50; padding: 2px 4px;"
        )
        self.left_panel_layout.addWidget(self.config_summary_title)

        self.config_summary_browser = QtWidgets.QTextBrowser(self.left_panel)
        self.config_summary_browser.setMinimumHeight(132)
        self.config_summary_browser.setMaximumHeight(148)
        self.config_summary_browser.setStyleSheet(
            "background-color: #eef3f8; "
            "color: #2c3e50; "
            "border: 1px solid #cfd8e3; "
            "border-radius: 4px; "
            "font-family: Consolas, 'Courier New', monospace; "
            "font-size: 13px; "
            "padding: 6px; "
            "line-height: 1.35;"
        )
        self.left_panel_layout.addWidget(self.config_summary_browser)

        self.last_run_title = QtWidgets.QLabel("最近一次运行结果")
        self.last_run_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2c3e50; padding: 2px 4px;"
        )
        self.left_panel_layout.addWidget(self.last_run_title)

        self.last_run_browser = QtWidgets.QTextBrowser(self.left_panel)
        self.last_run_browser.setMinimumHeight(96)
        self.last_run_browser.setMaximumHeight(132)
        self.last_run_browser.setStyleSheet(
            "background-color: #f5f9f4; "
            "color: #2c3e50; "
            "border: 1px solid #c8dcc4; "
            "border-radius: 4px; "
            "font-family: Consolas, 'Courier New', monospace; "
            "font-size: 12px; "
            "padding: 6px; "
            "line-height: 1.35;"
        )
        self.left_panel_layout.addWidget(self.last_run_browser)

        self.log_title = QtWidgets.QLabel("运行日志 / 数值输出")
        self.log_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2c3e50; padding: 2px 4px;"
        )
        self.left_panel_layout.addWidget(self.log_title)

        self.text_browser = QtWidgets.QTextBrowser(self.left_panel)
        self.text_browser.setMinimumWidth(350)
        self.text_browser.setStyleSheet(
            "background-color: #f8f9fa; "
            "color: #2c3e50; "
            "border: 1px solid #dcdde1; "
            "border-radius: 4px; "
            "font-family: Consolas, 'Courier New', monospace; "
            "font-size: 17px; "
            "padding: 8px; "
            "line-height: 1.5;"
        )
        self.left_panel_layout.addWidget(self.text_browser, 1)

        # --- 右下侧：图片内嵌大画板 ---
        self.canvas_container = QtWidgets.QWidget(self.splitter)
        self.canvas_container.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        # 按比例 30:70 分配左右，实际像素由 main.py 在 show() 后根据窗口宽度重设
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 7)
        self.splitter.setSizes([390, 910])
        self.main_vbox.addWidget(self.splitter, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
