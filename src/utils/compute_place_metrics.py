import numpy as np
from sklearn.neighbors import BallTree

# 위경도를 라디안으로 변환
def to_radians(df, lat_col='latitude', lon_col='longitude'):
    df['latitude_radi'] = np.radians(df[lat_col])
    df['longitude_radi'] = np.radians(df[lon_col])
    return df


def calculate_nearby_stats(main_df, places_dict, radii):
    """
    main_df: 원본 데이터 (계산 대상 위치)
    places_dict: 장소 데이터 딕셔너리, {장소 이름: 장소 데이터프레임}
    radii: 반경 리스트 (예: [300, 500, 1000])
    """
    
    # 위경도를 라디안으로 변환 (main_df에 적용)
    main_df = to_radians(main_df)

    # 각 장소 유형별로 BallTree를 생성하여 반경 내 개수를 계산
    for place_name, add_df in places_dict.items():

        # 각 장소 데이터의 위경도를 라디안으로 변환
        add_df = to_radians(add_df)

        # 장소 데이터에 대해 BallTree 생성
        ball_tree = BallTree(add_df[['latitude_radi', 'longitude_radi']].values, metric='haversine')

        for radius in radii:
            # 반경을 km로 변환
            radius_in_km = radius / 1000

            # 반경 내 place 개수 계산(반경을 지구 반지름(6371km)로 나눈 값)
            _, indices = ball_tree.query_radius(main_df[['latitude_radi', 'longitude_radi']].values, r=radius_in_km/6371, return_distance=True)

            # 반경별 장소 개수 열 추가
            main_df[f'{place_name}_{radius}m'] = [len(idx) for idx in indices]
        
        distances, distances_index = ball_tree.query(main_df[['latitude_radi', 'longitude_radi']].values, k=1)
        main_df[f'{place_name}_near_distance'] = distances.flatten()*6371

        # 공원: area 추가
        if place_name == 'park':
            main_df['park_near_area'] = add_df.iloc[distances_index.flatten(), 2].reset_index(drop=True)

    return main_df
