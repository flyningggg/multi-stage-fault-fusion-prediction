# target-screening-mvp-v1 运行记录

## 结论

- 状态：`completed`。
- 正式主流程已收敛为：精确拓扑分析 → 跨期一对一匹配 → 透明评分 → 限径空间聚合 → 稳定性分级 → 证据卡与图件导出。
- 最终真实三期结果：9872个节点全部进入匹配核算，形成6392个空间匹配单元；3480个两期支持单元满足候选资格，筛出696个候选单元，聚合为59个空间候选组。
- 其中9个达到稳定性门槛（一级6个、二级3个），50个明确标记为“不稳定候选”，仅供复核。
- 外部验证为`not_validated`；本结果只能支持内部勘探有利区辅助筛选，不能声称油气命中率。

## 可复现命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_target_screening.py --output program\data\multiperiod_results\target_screening_reproduction
```

首次完整精确运行耗时676.8秒；最终修复后的匹配、聚合与导出曾在本地复用相同配置下的精确时期指标，耗时45.8秒。仓库不提交该临时缓存，以上命令会从原始三期数据完整重算，不依赖开发机路径。`final/manifest.json`保留最终审计运行的环境记录与精确指标配置哈希；任何影响构图、距离变换或中心性定义的配置变化都会拒绝缓存复用。

## 验证证据

- 全量测试：100 passed，零失败。
- GUI无界面启动：`GUI_SMOKE_OK`，菜单包含“一键生成候选勘探有利区”“专业分析与图件”“研究与实验”。
- GUI真实Windows渲染：主操作卡和底部运行信息均可见；14×9英寸图在自适应模式缩放为616×396，未超过1178×460视口；原始模式恢复1400×900并启用滚动。
- 取消任务使用协作式中断，不再调用`QThread.terminate()`；启动说明改为非模态摘要。
- 原有绘图方法保留：原图、分类图、热力图、方位图、玫瑰图、三元图、关系图、拓扑图、轮廓图。
- 正式GUI与CLI均调用`screening_pipeline.run_target_screening`；正式路径不导入`agent_model`。
- 节点守恒：9872/9872；同一期重复匹配数：0。
- GeoPackage包含`all_candidate_targets`与`stable_candidate_targets`两个图层。
- `pip check`：No broken requirements found。
- `git diff --check`：无空白错误。

## 失败审计与修复

1. 首次DBSCAN跨期匹配产生49个链式大簇，累计丢弃8330个节点；该次3靶区结论已作废。改为一对一、每时期至多一个节点、簇内任意两点不超过容差的确定性匹配。
2. 第二次靶区DBSCAN使单个区域吞并621/696候选单元；该结果已作废。增加18km最大直径的完全链接拆分后，最大靶区23个单元。
3. 中间结果曾把低稳定率候选包装为一级；现规定稳定率低于0.80只能标记“不稳定候选”。
4. 三期网络均无割点，节点移除连通分量影响项在本数据上无区分力；综合分保留原合同但明确降低有效上限，不临时改权重粉饰结果。

## 产物

- `final/result.json`：完整稳定合同与结果。
- `final/stable_candidate_targets.csv`：9个稳定候选。
- `final/candidate_targets.csv`：全部59个候选空间组。
- `final/candidate_targets.gpkg`：GIS图层。
- `final/evidence_cards.json`：逐靶区证据卡。
- `final/maps/candidate_targets.png`：稳定候选重点图。
- `final/report.md`、`final/manifest.json`、`final/config_snapshot.yaml`：报告、环境清单和配置快照。

## 已知边界

- 无井位、产量、储层或专家盲评标签，外部有效性未验证。
- 源CSV未声明CRS；GeoPackage保留原始米制坐标但不写入EPSG，对接GIS前必须人工确认坐标参考系。
- 9个“稳定候选”仅表示对三种路径距离定义较稳定，不等同于商业油气发现概率。
