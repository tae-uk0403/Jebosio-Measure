

def position_info(cloth_cat_id):
    # ================================== 
    # annos['annotations']
    # ================================== 
    # * index for clothes category 
    # 1: short sleeve top 
    # 2: long sleeve top 
    # 7: shorts 
    # 8: Trousers 
    # 9: Skirt 
    # ... etc. 
    # ================================== 
    
    if cloth_cat_id == 1: # 1: short sleeve top 
        # *** T-Shirts ***
        # ---------------------------- 
        measure_points = {
            'Main' : (-180, 50), # TEXT
            'Total-length': (220, 50),   # 총장 
            'Chest-size': (-180, 120),  # 가슴둘레 # beside chest 
            'Waist-length': (230,50) ,  # 허리둘레
            'Arm-width': (-180, 120),  # 팔둘레
            'Shoulder-length': (-180, 120)   # 어깨길이
        }
    elif cloth_cat_id == 7: # 7: shorts 
        # *** Shorts ***
        # ---------------------------- 
        measure_points = {
            'Main' : (-100, -30), # TEXT
            'Waist-length' : (-50, 100) ,     # 허리둘레
            'Pants-length' : (100, 50) ,     # 총장 (=바지길이: pants length) 
            'Hole-width' : (250, -30) ,      # 밑단 
            'rise' : (30,-30)   # 밑위 ( 잘 안됨 ; 미정 )
            # 'e': 'Thigh-width',      # 허벅지 둘레 (: 미정)
        }

    return measure_points