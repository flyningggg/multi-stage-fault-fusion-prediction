import os
import geopandas as gpd


def geojson2shp(geojson_file, shp_file):
    """
    将geojson格式的文件转化为shapefile
    :param geojson_file: 需要转换的geojson文件名
    :param shp_file: 转换输出的shapefile文件名
    """

    if os.path.exists(shp_file):
        os.remove(shp_file)

    out_data = gpd.read_file(geojson_file)
    # if out_data.crs is None:
    #     out_data.crs = 'EPSG:4326'              # 无坐标的文件需要指定空间参考为4326（WGS84坐标）
    out_data.to_file(shp_file, driver='ESRI Shapefile', encoding='utf-8')


if __name__ == "__main__":
    trace_url = 'dataset/traces_200k.geojson'
    out_shp_url = 'dataset/json2shp3.shp'

    geojson2shp(trace_url, out_shp_url)
