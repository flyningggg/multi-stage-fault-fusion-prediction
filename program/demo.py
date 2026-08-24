# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 900)
        MainWindow.setWindowTitle("多期断裂网络勘探有利区辅助筛选系统 v2.3")
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # 主布局 (垂直)
        self.main_vbox = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_vbox.setContentsMargins(10, 10, 10, 10)

        # ==========================================
        # 上部：工具栏
        # ==========================================
        self.toolbar_layout = QtWidgets.QHBoxLayout()
        self.toolbar_layout.setSpacing(8)

        self.btn_open_model_dir = QtWidgets.QPushButton("打开 model/")
        self.btn_open_model_dir.setToolTip("打开当前程序目录下的 model 文件夹")
        self.btn_open_processed_dir = QtWidgets.QPushButton("打开 data/processed/")
        self.btn_open_processed_dir.setToolTip("打开导出的结果目录")
        self.btn_cancel_task = QtWidgets.QPushButton("取消任务")
        self.btn_cancel_task.setToolTip("取消正在后台执行的长任务")
        self.btn_export_results = QtWidgets.QPushButton("导出结果")
        self.btn_export_results.setToolTip("汇总显示最近一次运行的导出文件路径")
        self.btn_toggle_run_info = QtWidgets.QPushButton("隐藏运行信息")
        self.btn_toggle_run_info.setToolTip("显示或收起底部结果摘要与运行日志")

        _toolbar_btn_style = "QPushButton { font-size: 12px; padding: 4px 10px; }"
        for btn in [self.btn_open_model_dir, self.btn_open_processed_dir,
                    self.btn_cancel_task, self.btn_export_results, self.btn_toggle_run_info]:
            btn.setMinimumHeight(30)
            btn.setMaximumHeight(34)
            btn.setStyleSheet(_toolbar_btn_style)
            self.toolbar_layout.addWidget(btn)

        self.toolbar_layout.addStretch()
        self.main_vbox.addLayout(self.toolbar_layout)

        # ==========================================
        # 正式主流程入口：全局唯一主操作
        # ==========================================
        self.screening_banner = QtWidgets.QFrame()
        self.screening_banner.setObjectName("screeningBanner")
        self.screening_banner.setStyleSheet("""
            QFrame#screeningBanner {
                background-color: #e9f1f0;
                border: 1px solid #b8cecb;
                border-radius: 10px;
            }
            QLabel#screeningEyebrow {
                color: #526b68;
                font-size: 11px;
                font-weight: bold;
            }
            QLabel#screeningTitle {
                color: #203b3a;
                font-size: 18px;
                font-weight: bold;
            }
            QLabel#screeningSubtitle {
                color: #607472;
                font-size: 12px;
            }
            QLabel#screeningStatus {
                background-color: #f7faf9;
                color: #4c6663;
                border: 1px solid #c7d7d4;
                border-radius: 11px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton#primaryScreeningButton {
                background-color: #356f70;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 22px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton#primaryScreeningButton:hover { background-color: #2d6162; }
            QPushButton#primaryScreeningButton:pressed { background-color: #264f50; }
            QPushButton#primaryScreeningButton:disabled { background-color: #9fb4b2; }
            QPushButton#evidenceOverviewButton {
                background-color: #f7faf9;
                color: #315b59;
                border: 1px solid #9fbab6;
                border-radius: 7px;
                padding: 9px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton#evidenceOverviewButton:hover {
                background-color: #dfecea;
                border-color: #6f9994;
            }
            QPushButton#evidenceOverviewButton:pressed { background-color: #d2e2df; }
        """)
        self.screening_banner_layout = QtWidgets.QHBoxLayout(self.screening_banner)
        self.screening_banner_layout.setContentsMargins(18, 10, 14, 10)
        self.screening_banner_layout.setSpacing(14)
        self.screening_text_layout = QtWidgets.QVBoxLayout()
        self.screening_text_layout.setSpacing(1)
        self.screening_eyebrow = QtWidgets.QLabel("正式分析主流程")
        self.screening_eyebrow.setObjectName("screeningEyebrow")
        self.screening_title = QtWidgets.QLabel("多期断裂网络候选勘探有利区筛选")
        self.screening_title.setObjectName("screeningTitle")
        self.screening_subtitle = QtWidgets.QLabel(
            "精确拓扑 · 跨期匹配 · 稳定性分级 · 证据卡与GIS图层导出"
        )
        self.screening_subtitle.setObjectName("screeningSubtitle")
        self.screening_text_layout.addWidget(self.screening_eyebrow)
        self.screening_text_layout.addWidget(self.screening_title)
        self.screening_text_layout.addWidget(self.screening_subtitle)
        self.screening_banner_layout.addLayout(self.screening_text_layout, 1)
        self.screening_status_label = QtWidgets.QLabel("尚未运行")
        self.screening_status_label.setObjectName("screeningStatus")
        self.screening_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.btn_evidence_overview = QtWidgets.QPushButton("证据与数据状态")
        self.btn_evidence_overview.setObjectName("evidenceOverviewButton")
        self.btn_evidence_overview.setMinimumHeight(44)
        self.btn_evidence_overview.setMinimumWidth(150)
        self.btn_evidence_overview.setToolTip("查看已完成的 P2/P3 证据、主张边界和仍缺少的数据")
        self.btn_primary_screening = QtWidgets.QPushButton("生成候选勘探有利区  →")
        self.btn_primary_screening.setObjectName("primaryScreeningButton")
        self.btn_primary_screening.setMinimumHeight(44)
        self.btn_primary_screening.setMinimumWidth(235)
        self.btn_primary_screening.setToolTip("运行正式精确筛选流程；长任务可通过顶部“取消任务”停止")
        self.screening_banner_layout.addWidget(self.screening_status_label)
        self.screening_banner_layout.addWidget(self.btn_evidence_overview)
        self.screening_banner_layout.addWidget(self.btn_primary_screening)
        self.main_vbox.addWidget(self.screening_banner)

        # ==========================================
        # 中部：左右分栏结构
        # ==========================================
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # --- 左侧面板 ---
        self.left_panel = QtWidgets.QWidget(self.splitter)
        self.left_panel_layout = QtWidgets.QVBoxLayout(self.left_panel)
        self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.left_panel_layout.setSpacing(8)

        # --- 数据源选择 ---
        self.data_source_group = QtWidgets.QGroupBox("数据源")
        self.data_source_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #bdc3c7; border-radius: 4px; margin-top: 8px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        self.data_source_layout = QtWidgets.QVBoxLayout(self.data_source_group)

        self.lbl_data_source = QtWidgets.QLabel("选择数据源:")
        self.combo_data_source = QtWidgets.QComboBox()
        self.combo_data_source.addItems(["三期断裂数据"])
        self.combo_data_source.setMinimumWidth(160)
        self.combo_data_source.setMinimumHeight(30)

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
        self.combo_data_source.setStyleSheet(_combo_style)

        self.data_source_layout.addWidget(self.lbl_data_source)
        self.data_source_layout.addWidget(self.combo_data_source)
        self.left_panel_layout.addWidget(self.data_source_group)

        # --- 时期选择器 ---
        self.period_group = QtWidgets.QGroupBox("分析时期")
        self.period_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #bdc3c7; border-radius: 4px; margin-top: 8px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        self.period_layout = QtWidgets.QVBoxLayout(self.period_group)

        self.chk_period_haixi = QtWidgets.QCheckBox("海西期 (~4亿-2.5亿年前)")
        self.chk_period_xishan = QtWidgets.QCheckBox("喜山期 (~6500万年前至今)")
        self.chk_period_yinzhi = QtWidgets.QCheckBox("印支燕山期 (~2.5亿-1亿年前)")

        # 默认全部选中
        self.chk_period_haixi.setChecked(True)
        self.chk_period_xishan.setChecked(True)
        self.chk_period_yinzhi.setChecked(True)

        _chk_style = "QCheckBox { font-size: 13px; padding: 4px 0; }"
        for chk in [self.chk_period_haixi, self.chk_period_xishan, self.chk_period_yinzhi]:
            chk.setStyleSheet(_chk_style)
            self.period_layout.addWidget(chk)

        # 全选/全不选按钮
        self.period_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_select_all_periods = QtWidgets.QPushButton("全选")
        self.btn_deselect_all_periods = QtWidgets.QPushButton("全不选")
        _period_btn_style = "QPushButton { font-size: 11px; padding: 2px 6px; }"
        for btn in [self.btn_select_all_periods, self.btn_deselect_all_periods]:
            btn.setStyleSheet(_period_btn_style)
            self.period_btn_layout.addWidget(btn)
        self.period_layout.addLayout(self.period_btn_layout)

        self.left_panel_layout.addWidget(self.period_group)

        # --- 参数配置 ---
        self.params_group = QtWidgets.QGroupBox("运行参数")
        self.params_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #bdc3c7; border-radius: 4px; margin-top: 8px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        self.params_layout = QtWidgets.QFormLayout(self.params_group)
        self.params_layout.setSpacing(6)

        self.spin_kmeans_k = QtWidgets.QSpinBox()
        self.spin_kmeans_k.setRange(2, 24)
        self.spin_kmeans_k.setValue(4)
        self.spin_kmeans_k.setMinimumHeight(28)

        self.combo_train_target = QtWidgets.QComboBox()
        self.combo_train_target.setEditable(True)
        self.combo_train_target.setMinimumWidth(180)
        self.combo_train_target.setMinimumHeight(28)
        self.combo_train_target.setStyleSheet(_combo_style)

        self.dspin_grid_step = QtWidgets.QDoubleSpinBox()
        self.dspin_grid_step.setRange(10.0, 50000.0)
        self.dspin_grid_step.setDecimals(1)
        self.dspin_grid_step.setValue(3000.0)
        self.dspin_grid_step.setMinimumHeight(28)

        self.params_layout.addRow("聚类 k:", self.spin_kmeans_k)
        self.params_layout.addRow("训练目标列:", self.combo_train_target)
        self.params_layout.addRow("网格步长(m):", self.dspin_grid_step)

        self.left_panel_layout.addWidget(self.params_group)

        # --- 配置摘要 ---
        self.config_summary_title = QtWidgets.QLabel("当前配置摘要")
        self.config_summary_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2c3e50; padding: 2px 4px;"
        )
        self.left_panel_layout.addWidget(self.config_summary_title)

        self.config_summary_browser = QtWidgets.QTextBrowser(self.left_panel)
        self.config_summary_browser.setMinimumHeight(100)
        self.config_summary_browser.setMaximumHeight(120)
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

        self.left_panel_layout.addStretch(1)

        # --- 右侧面板：选项卡 + 画布 ---
        self.right_panel = QtWidgets.QWidget(self.splitter)
        self.right_panel_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.right_panel_layout.setSpacing(4)

        # ==========================================
        # 选项卡区域
        # ==========================================
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                color: #2c3e50;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #d5dbdb;
            }
        """)

        # --- Tab 1: 基础（拓扑绘图）---
        self.tab_basic = QtWidgets.QWidget()
        self.tab_basic_layout = QtWidgets.QVBoxLayout(self.tab_basic)
        self.tab_basic_layout.setSpacing(6)
        self.tab_basic_layout.setContentsMargins(8, 8, 8, 8)

        # 基础绘图按钮（3行）
        self.basic_row1_layout = QtWidgets.QHBoxLayout()
        self.btn_yuantu = QtWidgets.QPushButton("原断裂数据地图")
        self.btn_fenleihou = QtWidgets.QPushButton("分类后数据地图")
        self.btn_relitu = QtWidgets.QPushButton("断裂密度热力图")
        self.btn_azimuth = QtWidgets.QPushButton("方位角集图")
        self.btn_meiguitu = QtWidgets.QPushButton("方向玫瑰图")

        self.btn_yuantu.setToolTip("在研究区范围内绘制原始断裂迹线")
        self.btn_fenleihou.setToolTip("按 fractopo 规则对迹线做分支类型着色")
        self.btn_relitu.setToolTip("沿迹线采样后做核密度估计")
        self.btn_azimuth.setToolTip("按设定的方位组给迹线着色")
        self.btn_meiguitu.setToolTip("迹线/分支方位角的极坐标玫瑰图")

        _row_btn_style = "QPushButton { font-size: 12px; padding: 4px 6px; }"
        _sp_expand = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        for btn in [self.btn_yuantu, self.btn_fenleihou, self.btn_relitu, self.btn_azimuth, self.btn_meiguitu]:
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)
            btn.setSizePolicy(_sp_expand)
            btn.setStyleSheet(_row_btn_style)
            self.basic_row1_layout.addWidget(btn)
        self.tab_basic_layout.addLayout(self.basic_row1_layout)

        self.basic_row2_layout = QtWidgets.QHBoxLayout()
        self.btn_sanyuantu = QtWidgets.QPushButton("各类别三元图")
        self.btn_guanxi = QtWidgets.QPushButton("交叉与相邻关系")
        self.btn_b = QtWidgets.QPushButton("确定分支和节点")
        self.btn_a = QtWidgets.QPushButton("长度分布拟合")

        self.btn_sanyuantu.setToolTip("节点类型或分支端点组合的比例三角图")
        self.btn_guanxi.setToolTip("方位集之间的交叉与相邻关系示意")
        self.btn_b.setToolTip("迹线+研究区与节点类型、分支类型对比")
        self.btn_a.setToolTip("迹线与分支长度直方图 + 幂律/对数正态拟合")

        for btn in [self.btn_sanyuantu, self.btn_guanxi, self.btn_b, self.btn_a]:
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)
            btn.setSizePolicy(_sp_expand)
            btn.setStyleSheet(_row_btn_style)
            self.basic_row2_layout.addWidget(btn)
        self.tab_basic_layout.addLayout(self.basic_row2_layout)

        self.basic_row3_layout = QtWidgets.QHBoxLayout()
        self.combo_topo = QtWidgets.QComboBox()
        self.combo_topo.addItems(["请选择拓扑视图...", "拓扑化后断裂数据地图1", "拓扑化后断裂数据地图2"])
        self.combo_params = QtWidgets.QComboBox()
        self.combo_params.addItems(["下拉选择绘制参数...", "Fracture Intensity B21 / P21 (断裂强度)",
                                    "Trace Max/Min/Mean Length (迹线长度分布)", "Dimensionless Intensity (无量纲强度)",
                                    "Number of Traces (实际迹线数量)", "Branch Max/Min/Mean Length", "Node Count",
                                    "Branch Count"])
        self.btn_tuopushuxing = QtWidgets.QPushButton("显示拓扑属性数据")

        self.combo_topo.setStyleSheet(_combo_style)
        self.combo_params.setStyleSheet(_combo_style)
        self.btn_tuopushuxing.setStyleSheet(_row_btn_style)

        for widget in [self.combo_topo, self.combo_params, self.btn_tuopushuxing]:
            widget.setMinimumHeight(32)
            widget.setMaximumHeight(36)
            self.basic_row3_layout.addWidget(widget)
        self.basic_row3_layout.addStretch()
        self.tab_basic_layout.addLayout(self.basic_row3_layout)

        self.tab_basic_layout.addStretch()
        self.tab_widget.addTab(self.tab_basic, "基础")

        # --- Tab 2: 融合（属性融合分析）---
        self.tab_fusion = QtWidgets.QWidget()
        self.tab_fusion_layout = QtWidgets.QVBoxLayout(self.tab_fusion)
        self.tab_fusion_layout.setSpacing(6)
        self.tab_fusion_layout.setContentsMargins(8, 8, 8, 8)

        self.fusion_row1_layout = QtWidgets.QHBoxLayout()
        self.combo_fusion = QtWidgets.QComboBox()
        self.combo_fusion.addItems(["融合方法: PCA", "融合方法: 自编码器", "融合方法: UMAP", "融合方法: VAE"])
        self.combo_fusion.setStyleSheet(_combo_style)
        self.combo_fusion.setMaximumWidth(180)
        self.btn_ronghe = QtWidgets.QPushButton("执行属性融合分析")
        self.btn_guoji_weighted = QtWidgets.QPushButton("高价值属性加权融合")
        self.btn_guoji_compare = QtWidgets.QPushButton("融合对比(加权vsGAT)")
        self.btn_k_helper = QtWidgets.QPushButton("选k辅助")

        self.combo_fusion.setToolTip("将多列网格拓扑属性降维到2维，再做KMeans聚类")
        self.btn_ronghe.setToolTip("读取当前工区网格CSV，按所选方法做属性融合+聚类")
        self.btn_guoji_weighted.setToolTip("对高价值连通类属性加权，得到每网格一维得分")
        self.btn_guoji_compare.setToolTip("对比规则加权融合与GAT图网络融合得分分布")
        self.btn_k_helper.setToolTip("基于当前融合特征计算不同k的曲线，辅助选择聚类数")

        for widget in [self.combo_fusion, self.btn_ronghe, self.btn_guoji_weighted, self.btn_guoji_compare, self.btn_k_helper]:
            widget.setMinimumHeight(32)
            widget.setMaximumHeight(36)
            if isinstance(widget, QtWidgets.QPushButton):
                widget.setSizePolicy(_sp_expand)
                widget.setStyleSheet(_row_btn_style)
            self.fusion_row1_layout.addWidget(widget)
        self.tab_fusion_layout.addLayout(self.fusion_row1_layout)

        self.fusion_row2_layout = QtWidgets.QHBoxLayout()
        self.btn_guoji_train = QtWidgets.QPushButton("训练XGBoost模型")
        self.lbl_shap_features = QtWidgets.QLabel("SHAP关注特征:")
        self.combo_shap_features = QtWidgets.QComboBox()
        self.combo_shap_features.setMinimumWidth(180)
        self.combo_shap_features.setMaximumWidth(220)
        self.combo_shap_features.setStyleSheet(_combo_style)
        self.combo_shap_features.addItem("全部（默认顺序）")
        self.btn_guoji_shap = QtWidgets.QPushButton("SHAP可解释分析")
        self.btn_spatial = QtWidgets.QPushButton("一键空间-拓扑融合")

        self.btn_guoji_train.setToolTip("用特征工程后的矩阵训练XGBoost")
        self.btn_guoji_shap.setToolTip("对已训练模型做SHAP特征重要性")
        self.btn_spatial.setToolTip("特征工程→加权融合→GAT→XGBoost→SHAP一键跑完")

        for widget in [self.btn_guoji_train, self.lbl_shap_features, self.combo_shap_features, self.btn_guoji_shap, self.btn_spatial]:
            widget.setMinimumHeight(32)
            widget.setMaximumHeight(36)
            if isinstance(widget, QtWidgets.QPushButton):
                widget.setSizePolicy(_sp_expand)
                widget.setStyleSheet(_row_btn_style)
            self.fusion_row2_layout.addWidget(widget)
        self.tab_fusion_layout.addLayout(self.fusion_row2_layout)

        self.tab_fusion_layout.addStretch()
        self.tab_widget.addTab(self.tab_fusion, "融合")

        # --- Tab 3: 渗流（图渗流模拟）---
        self.tab_percolation = QtWidgets.QWidget()
        self.tab_percolation_layout = QtWidgets.QVBoxLayout(self.tab_percolation)
        self.tab_percolation_layout.setSpacing(6)
        self.tab_percolation_layout.setContentsMargins(8, 8, 8, 8)

        self.percolation_row1_layout = QtWidgets.QHBoxLayout()
        self.btn_percolation_curves = QtWidgets.QPushButton("渗流曲线对比")
        self.btn_key_nodes = QtWidgets.QPushButton("关键节点图")
        self.btn_boundary_analysis = QtWidgets.QPushButton("边界效应分析")

        self.btn_percolation_curves.setToolTip("对比三期的渗流曲线（最大连通分量大小 vs 删边占比）")
        self.btn_key_nodes.setToolTip("为每个选定时期生成关键节点空间分布图")
        self.btn_boundary_analysis.setToolTip("对比排除/不排除边界节点的结果")

        for btn in [self.btn_percolation_curves, self.btn_key_nodes, self.btn_boundary_analysis]:
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)
            btn.setSizePolicy(_sp_expand)
            btn.setStyleSheet(_row_btn_style)
            self.percolation_row1_layout.addWidget(btn)
        self.tab_percolation_layout.addLayout(self.percolation_row1_layout)

        self.tab_percolation_layout.addStretch()
        self.tab_widget.addTab(self.tab_percolation, "渗流")

        # --- Tab 4: 代理（代理模型）---
        self.tab_agent = QtWidgets.QWidget()
        self.tab_agent_layout = QtWidgets.QVBoxLayout(self.tab_agent)
        self.tab_agent_layout.setSpacing(6)
        self.tab_agent_layout.setContentsMargins(8, 8, 8, 8)

        self.agent_row1_layout = QtWidgets.QHBoxLayout()
        self.btn_agent_train = QtWidgets.QPushButton("训练代理模型")
        self.btn_agent_pred_vs_true = QtWidgets.QPushButton("预测vs真实散点图")
        self.btn_agent_shap = QtWidgets.QPushButton("SHAP蜂群图")
        self.btn_agent_importance = QtWidgets.QPushButton("特征重要性图")

        self.btn_agent_train.setToolTip("训练XGBoost代理模型，预测betweenness centrality")
        self.btn_agent_pred_vs_true.setToolTip("生成预测值vs真实值的散点图")
        self.btn_agent_shap.setToolTip("生成SHAP蜂群图，分析特征对预测的影响")
        self.btn_agent_importance.setToolTip("生成特征重要性柱状图")

        for btn in [self.btn_agent_train, self.btn_agent_pred_vs_true, self.btn_agent_shap, self.btn_agent_importance]:
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)
            btn.setSizePolicy(_sp_expand)
            btn.setStyleSheet(_row_btn_style)
            self.agent_row1_layout.addWidget(btn)
        self.tab_agent_layout.addLayout(self.agent_row1_layout)

        self.tab_agent_layout.addStretch()
        self.tab_widget.addTab(self.tab_agent, "代理")

        # --- Tab 5: 实验（对比实验）---
        self.tab_experiment = QtWidgets.QWidget()
        self.tab_experiment_layout = QtWidgets.QVBoxLayout(self.tab_experiment)
        self.tab_experiment_layout.setSpacing(6)
        self.tab_experiment_layout.setContentsMargins(8, 8, 8, 8)

        self.experiment_row1_layout = QtWidgets.QHBoxLayout()
        self.btn_exp_ablation = QtWidgets.QPushButton("消融实验")
        self.btn_exp_model_compare = QtWidgets.QPushButton("算法族对比")
        self.btn_exp_noise = QtWidgets.QPushButton("噪声敏感性")
        self.btn_exp_spatial_cv = QtWidgets.QPushButton("空间交叉验证")

        self.btn_exp_ablation.setToolTip("运行消融实验，对比不同特征组合的效果")
        self.btn_exp_model_compare.setToolTip("对比不同算法的表现")
        self.btn_exp_noise.setToolTip("测试模型对噪声的敏感性")
        self.btn_exp_spatial_cv.setToolTip("运行空间交叉验证")

        for btn in [self.btn_exp_ablation, self.btn_exp_model_compare, self.btn_exp_noise, self.btn_exp_spatial_cv]:
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)
            btn.setSizePolicy(_sp_expand)
            btn.setStyleSheet(_row_btn_style)
            self.experiment_row1_layout.addWidget(btn)
        self.tab_experiment_layout.addLayout(self.experiment_row1_layout)

        self.experiment_row2_layout = QtWidgets.QHBoxLayout()
        self.btn_exp_params_compare = QtWidgets.QPushButton("拓扑参数对比")
        self.btn_exp_length_powerlaw = QtWidgets.QPushButton("长度分布幂律")
        self.btn_exp_grid_sampling = QtWidgets.QPushButton("网格采样分析")

        self.btn_exp_params_compare.setToolTip("对比不同时期的拓扑参数")
        self.btn_exp_length_powerlaw.setToolTip("分析迹线长度的幂律分布")
        self.btn_exp_grid_sampling.setToolTip("运行网格采样分析")

        for btn in [self.btn_exp_params_compare, self.btn_exp_length_powerlaw, self.btn_exp_grid_sampling]:
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)
            btn.setSizePolicy(_sp_expand)
            btn.setStyleSheet(_row_btn_style)
            self.experiment_row2_layout.addWidget(btn)
        self.tab_experiment_layout.addLayout(self.experiment_row2_layout)

        self.tab_experiment_layout.addStretch()
        self.tab_widget.addTab(self.tab_experiment, "实验")

        # 约束选项卡高度：防止 tab 内容抢占画布空间导致与下方显示区重叠
        # tab 内仅含按钮行，200px 足够 3 行 + 间距；超出不影响布局
        self.tab_widget.setMaximumHeight(200)
        self.right_panel_layout.addWidget(self.tab_widget)

        # ==========================================
        # 画布区域
        # ==========================================
        # canvas_layout 仍是直接持有 gallery_control_layout 的容器，
        # 但实际画布（canvas_display_layout）渲染到 scroll_area 内，
        # 这样 matplotlib figure 即使尺寸超过可用空间也能滚动而不撑破 VBox
        self.canvas_container = QtWidgets.QWidget()
        self.canvas_container.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_layout.setSpacing(0)

        # 画布滚动容器：figure 物理尺寸大于可用区时可滚动，杜绝反向撑破布局
        self.canvas_scroll = QtWidgets.QScrollArea()
        self.canvas_scroll.setWidgetResizable(True)
        self.canvas_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.canvas_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.canvas_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.canvas_scroll.setStyleSheet("QScrollArea { background-color: white; }")
        # 滚动区内部的承载控件，canvas_display_layout 将挂在其上
        self.canvas_scroll_content = QtWidgets.QWidget()
        self.canvas_scroll_content.setStyleSheet("background-color: white;")
        self.canvas_scroll.setWidget(self.canvas_scroll_content)

        # 翻页控制
        self.gallery_control_layout = QtWidgets.QHBoxLayout()
        self.btn_prev_fig = QtWidgets.QPushButton("◀ 上一张")
        self.btn_next_fig = QtWidgets.QPushButton("下一张 ▶")
        self.lbl_fig_status = QtWidgets.QLabel("第1张/共1张")
        self.combo_figure_view_mode = QtWidgets.QComboBox()
        self.combo_figure_view_mode.setObjectName("figureViewMode")
        self.combo_figure_view_mode.addItem("清晰适配", "smart")
        self.combo_figure_view_mode.addItem("完整显示", "fit")
        self.combo_figure_view_mode.addItem("适应宽度", "width")
        self.combo_figure_view_mode.addItem("原始尺寸", "original")
        self.combo_figure_view_mode.setMinimumWidth(112)
        self.combo_figure_view_mode.setToolTip(
            "清晰适配会适度放大并只保留少量纵向滚动；也可完整显示、适应宽度或查看原始尺寸"
        )
        self.combo_figure_view_mode.setStyleSheet("""
            QComboBox#figureViewMode {
                color: #315b59; background: #f7faf9; border: 1px solid #9fbab6;
                border-radius: 5px; padding: 4px 9px; font-size: 12px;
            }
            QComboBox#figureViewMode:hover { background: #e7f0ee; }
            QComboBox#figureViewMode::drop-down { border: none; width: 22px; }
        """)
        self.btn_focus_fig = QtWidgets.QPushButton("专注查看")
        self.btn_focus_fig.setObjectName("figureFocusButton")
        self.btn_focus_fig.setCheckable(True)
        self.btn_focus_fig.setMinimumWidth(88)
        self.btn_focus_fig.setToolTip("临时隐藏参数区和运行信息，让当前图件占满主窗口；Esc 可退出")
        self.btn_focus_fig.setStyleSheet("""
            QPushButton#figureFocusButton {
                color: #405957; background: #f5f8f7; border: 1px solid #cbd8d6;
                border-radius: 5px; padding: 4px 10px; font-size: 12px;
            }
            QPushButton#figureFocusButton:checked {
                color: #ffffff; background: #547f7d; border-color: #547f7d;
            }
        """)
        self.lbl_fig_status.setAlignment(QtCore.Qt.AlignCenter)

        _gallery_btn_style = "QPushButton { font-size: 12px; padding: 4px 10px; }"
        self.btn_prev_fig.setStyleSheet(_gallery_btn_style)
        self.btn_next_fig.setStyleSheet(_gallery_btn_style)
        self.lbl_fig_status.setStyleSheet("font-size: 13px; color: #2c3e50;")

        self.gallery_control_layout.addWidget(self.combo_figure_view_mode)
        self.gallery_control_layout.addWidget(self.btn_focus_fig)
        self.gallery_control_layout.addStretch()
        self.gallery_control_layout.addWidget(self.btn_prev_fig)
        self.gallery_control_layout.addWidget(self.lbl_fig_status)
        self.gallery_control_layout.addWidget(self.btn_next_fig)
        self.gallery_control_layout.addStretch()

        self.canvas_layout.addLayout(self.gallery_control_layout)
        # 将画布滚动区放入画布容器（在翻页控制栏之下），占据剩余弹性空间
        self.canvas_layout.addWidget(self.canvas_scroll, 1)
        for _w in (self.btn_prev_fig, self.btn_next_fig, self.lbl_fig_status):
            _w.setVisible(False)
        # 初始化一个空布局挂到滚动区内容控件上，供 main.py.embed_figure 复用/清理
        # （embed_figure 会移除旧 canvas_display_layout 并新建，因此初始值非 None 即可）
        self.canvas_display_layout = QtWidgets.QVBoxLayout(self.canvas_scroll_content)
        self.canvas_display_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_display_layout.setSpacing(0)
        self.canvas_display_layout.addStretch()
        self.right_panel_layout.addWidget(self.canvas_container, 1)

        # 按比例 25:75 分配左右
        self.splitter.setStretchFactor(0, 25)
        self.splitter.setStretchFactor(1, 75)
        self.splitter.setSizes([350, 1050])
        self.main_vbox.addWidget(self.splitter, 1)

        # ==========================================
        # 次级运行信息：摘要与日志共享一个可收起区域
        # ==========================================
        self.run_info_tabs = QtWidgets.QTabWidget()
        self.run_info_tabs.setObjectName("runInfoTabs")
        self.run_info_tabs.setMinimumHeight(105)
        self.run_info_tabs.setMaximumHeight(155)
        self.run_info_tabs.setStyleSheet("""
            QTabWidget#runInfoTabs::pane {
                border: 1px solid #d5dddc;
                border-radius: 5px;
                background: #fafcfb;
            }
            QTabBar::tab {
                padding: 5px 16px;
                color: #526260;
                background: #edf2f1;
                border: 1px solid #d5dddc;
            }
            QTabBar::tab:selected {
                color: #244c4b;
                background: #ffffff;
                font-weight: bold;
            }
        """)
        self.run_summary_tab = QtWidgets.QWidget()
        self.run_summary_layout = QtWidgets.QVBoxLayout(self.run_summary_tab)
        self.run_summary_layout.setContentsMargins(6, 6, 6, 6)
        self.last_run_browser = QtWidgets.QTextBrowser()
        self.last_run_browser.setPlaceholderText("完成一次分析后，这里显示关键结果、证据边界和导出位置。")
        self.last_run_browser.setStyleSheet(
            "background: transparent; color: #2c3e50; border: none; "
            "font-family: 'Microsoft YaHei', sans-serif; font-size: 12px; padding: 2px;"
        )
        self.run_summary_layout.addWidget(self.last_run_browser)
        self.run_info_tabs.addTab(self.run_summary_tab, "结果摘要")

        self.run_log_tab = QtWidgets.QWidget()
        self.run_log_layout = QtWidgets.QVBoxLayout(self.run_log_tab)
        self.run_log_layout.setContentsMargins(6, 6, 6, 6)
        self.text_browser = QtWidgets.QTextBrowser()
        self.text_browser.setPlaceholderText("运行进度、诊断信息和错误详情会显示在这里。")
        self.text_browser.setStyleSheet(
            "background: transparent; color: #344443; border: none; "
            "font-family: Consolas, 'Courier New', monospace; font-size: 12px; padding: 2px;"
        )
        self.run_log_layout.addWidget(self.text_browser)
        self.run_info_tabs.addTab(self.run_log_tab, "运行日志")
        self.main_vbox.addWidget(self.run_info_tabs)

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
