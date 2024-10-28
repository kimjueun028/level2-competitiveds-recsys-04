# 🏠 수도권 아파트 전세가 예측 모델

## 📌 프로젝트 개요
본 프로젝트는 수도권 아파트 전세 실거래가를 예측하는 모델을 개발하는 것을 목표로 했습니다.  
시계열 데이터 분석, 머신러닝 모델링 등 다양한 기술을 실제 문제에 적용하여, 예측하는 것이 주요 과제였습니다.
## 📊 프로젝트 데이터
제공된 데이터셋은 아파트 전세 실거래가 예측을 목표로 하며, 학습 데이터와 테스트 데이터에는 각 건물의 기본적인 정보와 전세 실거래가로 구성됩니다.  
또한 지하철, 공원, 학교, 금리 데이터가 추가적으로 제공됩니다.

- **학습 데이터**: 2019년 4월 1일 ~ 2023년 12월 31일 (1,801,228개)
- **평가 데이터**: 2024년 1월 1일 ~ 2024년 6월 20일 (150,172개)
- **금리 데이터**: 2018년 12월 ~ 2024년 7월 (68개)
- **공원 데이터**: 위도, 경도, 면적 (17,564개)
- **학교 데이터**: 위도, 경도 (11,992개)
- **지하철 데이터**: 위도, 경도 (700개)
## 🗂️ 파일 구조
```
.
├── README.md
├── config
│   ├── cluster_dudep.pkl
│   ├── ensemble.yaml
│   ├── index_list.pkl
│   ├── lgbm_params.yaml
│   ├── radii_values.yaml
│   ├── radii_values_lgbm.yaml
│   └── xgb_params_info.pkl
├── data
├──src
    ├── ensemble.py
    ├── feature-extraction.py
    ├── models
    │   ├── lgbm_cluster.py
    │   ├── xgb_cluster.py
    │   ├── xgb_dedup_tuning.py
    │   └── xgb_valid_test.py
    ├── preprocessing
    │   ├── remove_duplicates.py
    │   └── xgb_dedup_grid_search.py
    └── utils
        ├── HuberLoss.py
        ├── age_group.py
        ├── compute_place_metrics.py
        └── config_loader.py
```

### 폴더 및 파일 설명
- **config 폴더**
 
    `raddi_values_lgbm.yaml` 은 `lgbm.py` 에 들어가는 데이터를 만들기 위한 파일입니다.

    `raddi_values.yaml` 은 `xgb_depoist.py`, `xgb_deposit_per_area.py` 에 들어가는 데이터를 만들기 위한 파일입니다.

    `index_list.pkl`, `cluster_dedep.pkl`, `xgb_params_info.pkl` 파일들은 `xgb_deposit_per_area.py` 를 실행할 때 사용하는 파일들 입니다.

    `ensemble.yaml` 은 `esemble.py` 를 실행할 때 사용하는 YAML 파일입니다. 
    앙상블하고 싶은 CSV 파일과 각 모델에 할당할 가중치가 적혀 있습니다.
  

- **results 폴더**

    `xgb_deposit_per_area.csv`, `xgb_deposit.csv`, `lgbm.csv` : 각 모델의 예측 결과가 저장된 파일들입니다.
    `esemble.csv` : 모델의 예측 결과로 앙상블 한 파일입니다.


- **preprocessing 폴더**
  
    `remove_duplicates.py` : train data에서 중복 제거를 위한 코드입니다.
    `xgb_dedup_grid_search.py` : `xgb_deposit_per.py`에 사용 하기 위한 hyper parameter를 grid search로 찾는 코드입니다.


- **src 폴더**

    `ensemble.py`: 여러 모델의 예측 결과를 soft voting 방식으로 앙상블해주는 코드입니다. YAML 파일을 읽어와 가중치와 함께 예측을 진행합니다.
    `feature-extraction.py` : 변수를 추가하는 코드입니다
    
- **models 폴더**

    `lgbm.py`: LightGBM 바탕으로 만든 모델입니다. 세부 내용은 [#9 PR](https://github.com/boostcampaitech7/level2-competitiveds-recsys-04/pull/9)에서 확인할 수 있습니다.

    `xgb_deposit_per_area.py`: XGBoost를 바탕으로 만든 모델입니다. 자세한 내용은 [#11 PR](https://github.com/boostcampaitech7/level2-competitiveds-recsys-04/pull/11)을 참고하세요.

    `xgb_deposit.py`: XGBoost 기반 모델로, 자세한 내용은  [#10 PR](https://github.com/boostcampaitech7/level2-competitiveds-recsys-04/pull/10)에서 확인할 수 있습니다.

- **Utils 폴더**

   `HuberLoss.py`: 모델을 돌리기 위한 custom loss, custom metric이 구현된 코드입니다.
  
   `age_group.py`: `feature-extraction.py`에 사용되는 건물의 연도로 특정 숫자를 부여합니다.
  
   `compute_place_metric.py`: `feature-extraction.py`에 사용되는 거리 계산을 해주는 코드입니다.
  
   `config_loader.py`: `feature-extraction.py`에 사용되는 코드입니다.
  

## 🛠️ 사용 방법
1. **개별 모델 실행:**  
   `src` 폴더에 존재하는 `feature-extraction`을 실행하면 모델 실행을 위한 데이터 파일(`train_aftercountplace.csv`, `test_aftercountplace.csv`)이 생성됩니다.
   
   실행 시, `config` 폴더에 있는 적용할 `radii_values` YAML 파일을 입력으로 제공해야 합니다.

    ```
    python feature-extraction.py radii_values.yaml
    ```

    이후 `src/models` 폴더에 존재하는 각 모델을 실행하면 예측 결과 파일(`xgb_deposit.csv`, `xgb_deposit_per_area.csv`, `lgbm.csv`)이 생성됩니다.

    ```
    python model_file_name.py
    ```
2. **앙상블(Ensemble) 실행:**  
    `src/ensemble.py`는 각 모델이 예측한 `csv` 파일을 읽어 가중치에 따른 앙상블을 진행합니다.  

    앙상블 실행 시, `config` 폴더에 있는 `ensemble.yaml` 파일을 읽으며, 해당 YAML 파일은 예측에 사용될 CSV 파일과 각 파일의 가중치를 정의합니다.  

    ```
    python ensemble.py 
    ```
    
## 🎯 파이널 제출 내역


- **제출 파일**: `results/Ensemble.csv`
- **Private MAE**: 4262.3147
- **최종 순위**: 3등

  ### 제출 파일 생성 방법
  파이널로 제출한 파일은 `ensemble.py` 스크립트를 실행하여 생성할 수 있습니다. 
  
  다음 명령어를 사용하여 `Ensemble.csv` 파일을 생성하세요:

  ```
  python ensemble.py
  ```


## 😊 팀 구성원
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/Heukma"><img src="https://avatars.githubusercontent.com/u/77618270?v=4" width="100px;" alt=""/><br /><sub><b>성효제</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/gagoory7"><img src="https://avatars.githubusercontent.com/u/163074222?v=4" width="100px;" alt=""/><br /><sub><b>백상민</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/Timeisfast"><img src="https://avatars.githubusercontent.com/u/120894109?v=4" width="100px;" alt=""/><br /><sub><b>김성윤</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/annakong23"><img src="https://avatars.githubusercontent.com/u/102771961?v=4" width="100px;" alt=""/><br /><sub><b>공지원</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/kimjueun028"><img src="https://avatars.githubusercontent.com/u/92249116?v=4" width="100px;" alt=""/><br /><sub><b>김주은</b></sub><br />
    </td>
    </td>
        <td align="center"><a href="https://github.com/zip-sa"><img src="https://avatars.githubusercontent.com/u/49730616?v=4" width="100px;" alt=""/><br /><sub><b>박승우</b></sub><br />
    </td>
  </tr>
</table>
</div>

<br />
