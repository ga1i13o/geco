import numpy as np
x=1

def part2ind(i):
    """
    Maps object parts to indices and generates keypoint permutations and orientations.
    This function takes an object category index `i` and returns a dict of parts names containing part indices as values (`pimap`),
    a permutation of parts (`PART_PERMUTATION`), and a boolean array indicating parts with orientation
    (`PART_WITH_ORIENTATION`). 
    Args:
        i (int): The object category index. Supported values correspond to specific object categories:
            1: Aeroplane
            2: Bicycle
            3: Bird
            4: Boat
            5: Bottle
            6, 7: Bus, Car
            8, 12: Cat, Dog
            9: Chair
            10, 13, 17: Cow, Horse, Sheep
            11: Table
            14: Motorbike
            15: Person
            16: Potted Plant
            18: Sofa
            19: Train
            20: TV Monitor
    Returns:
        tuple:
            - pimap (dict): A dictionary mapping part names to indices for the given object category.
            - PART_PERMUTATION (list or None): A list representing the permutation of parts for the object category.
              If the object category does not have enough parts, returns None.
            - PART_WITH_ORIENTATION (numpy.ndarray or None): A boolean array indicating parts with orientation.
              If the object category does not have enough parts, returns None.
    Notes:
        - The function handles special cases for certain object categories, such as additional parts or specific
          permutations.
        - For categories without parts or parts, `pimap` will be empty, and `PART_PERMUTATION` and `PART_WITH_ORIENTATION`
          will be None.
        - Not ethat the function includes logic to adjust permutations and orientations for valid parts only (excluding indices, that are not assigned to any part).
    """
    PART_PERMUTATION = None
    if i==1:
    # [aeroplane]
        pimap = {'body':1, 'stern':2, 'lwing':3, 'rwing':4, 'tail':5}
        for ii in range(1,11):
            pimap['engine_%d'%ii] = 10+ii
        for ii in range(1,11):
            pimap['wheel_%d'%ii] = 20+ii

        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()
        PART_PERMUTATION[3-x] = 4  # lwing
        PART_PERMUTATION[4-x] = 3  # rwing

    elif i==2:
    # [bicycle]
        pimap = {'fwheel':1, 'bwheel':2, 'saddle':3, 'handlebar':4, 'chainwheel':5}
        for ii in range(1,11):
            pimap['headlight_%d'%ii] = 10+ii

        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()
        PART_PERMUTATION[1-x] = 2  # fwheel
        PART_PERMUTATION[2-x] = 1  # bwheel
    
    elif i==3:
    # [bird]
        pimap = {'head':1, 'leye':2, 'reye':3, 'beak':4, 'torso':5, 'neck':6, 'lwing':7, 'rwing':8, 'lleg':9, 'lfoot':10, 'rleg':11, 'rfoot':12, 'tail':13}
        PART_PERMUTATION = [1,3,2,4,5,6,8,7,11,12,9,10,13]
    elif i==4:
    # [boat]
        pimap = {}
    elif i==5:
    # [bottle]
        pimap = {'cap':1, 'body':2}
        n_parts = max(pimap.values())

        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()

    elif i==6 or i==7:
    # [bus]
    # [car]
        pimap = {'frontside':1, 'leftside':2, 'rightside':3, 'backside':4, 'roofside':5, 'leftmirror':6, 'rightmirror':7, 'fliplate':8, 'bliplate':9}
        for ii in range(1,11):
            pimap['door_%d'%ii] = 10+ii
        for ii in range(1,11):
            pimap['wheel_%d'%ii] = 20+ii
        for ii in range(1,11):
            pimap['headlight_%d'%ii] = 30+ii
        for ii in range(1,21):
            pimap['window_%d'%ii] = 40+ii
        
        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()
        PART_PERMUTATION[1-x] = 4  # frontside
        PART_PERMUTATION[2-x] = 3 # leftside
        PART_PERMUTATION[3-x] = 2  # rightside
        PART_PERMUTATION[4-x] = 1  # backside
        PART_PERMUTATION[6-x] = 7  # leftmirror
        PART_PERMUTATION[7-x] = 6  # rightmirror
        PART_PERMUTATION[8-x] = 9  # fliplate
        PART_PERMUTATION[9-x] = 8  # bliplate

    elif i==8 or i==12:
    # [cat]
    # [dog]
        pimap = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'nose':6, 'torso':7, 'neck':8, 'lfleg':9, 'lfpa':10, 'rfleg':11, 'rfpa':12, 'lbleg':13, 'lbpa':14, 'rbleg':15, 'rbpa':16, 'tail':17}
        if i==12:
            pimap['muzzle'] = 18

        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()
        PART_PERMUTATION[2-x] = 3  # leye
        PART_PERMUTATION[3-x] = 2  # reye
        PART_PERMUTATION[4-x] = 5  # lear
        PART_PERMUTATION[5-x] = 4  # rear
        PART_PERMUTATION[9-x] = 10  # lfleg
        PART_PERMUTATION[10-x] = 9  # lfpa
        PART_PERMUTATION[11-x] = 12  # rfleg
        PART_PERMUTATION[12-x] = 11  # rfpa
        PART_PERMUTATION[13-x] = 15  # lbpa
        PART_PERMUTATION[15-x] = 13  # rbleg
        PART_PERMUTATION[14-x] = 16  # lbleg
        PART_PERMUTATION[16-x] = 14  # rbpa

    elif i==9:
    # [chair]
        pimap = {}
    elif i==10 or i==13 or i==17:
    # [cow]
    # [horse]
    # [sheep]
        pimap = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'muzzle':6, 'torso':9, 'neck':10, 'lfuleg':11, 'lflleg':12, 'rfuleg':13, 'rflleg':14, 'lbuleg':15, 'lblleg':16, 'rbuleg':17, 'rblleg':18, 'tail':19}
        if i==10 or i==17:
            # [cow]
            # [sheep]
            pimap['lhorn'] = 7
            pimap['rhorn'] = 8
        if i==13:
            # [horse]
            pimap['lfho'] = 30
            pimap['rfho'] = 31
            pimap['lbho'] = 32
            pimap['rbho'] = 33

        
        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()
        PART_PERMUTATION[2-x] = 3  # leye
        PART_PERMUTATION[3-x] = 2  # reye
        PART_PERMUTATION[4-x] = 5  # lear
        PART_PERMUTATION[5-x] = 4
        PART_PERMUTATION[11-x] = 13
        PART_PERMUTATION[13-x] = 11
        PART_PERMUTATION[12-x] = 14
        PART_PERMUTATION[14-x] = 12
        PART_PERMUTATION[15-x] = 17
        PART_PERMUTATION[17-x] = 15
        PART_PERMUTATION[16-x] = 18
        PART_PERMUTATION[18-x] = 16
        if i==10 or i==17:
            PART_PERMUTATION[7-x] = 8  # lhorn
            PART_PERMUTATION[8-x] = 7  # rhorn
        if i==13:
            PART_PERMUTATION[30-x] = 31
            PART_PERMUTATION[31-x] = 30
            PART_PERMUTATION[32-x] = 33
            PART_PERMUTATION[33-x] = 32

    elif i==11:
    # [table]
        pimap = {}
    elif i==14:
    # [motorbike]
        pimap = {'fwheel':1, 'bwheel':2, 'handlebar':3, 'saddle':4}
        for ii in range(1,11):
            pimap['headlight_%d'%ii] = 10+ii
 
        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()
        PART_PERMUTATION[1-x] = 2  # fwheel
        PART_PERMUTATION[2-x] = 1  # bwheel
        PART_PERMUTATION[11-x] = 12  # headlight
        PART_PERMUTATION[12-x] = 11  # headlight

    elif i==15:
    # [person]
        pimap = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'lebrow':6, 'rebrow':7, 'nose':8, 'mouth':9, 'hair':10, 'torso':11, 'neck':12, 'llarm':13, 'luarm':14, 'lhand':15, 'rlarm':16, 'ruarm':17, 'rhand':18, 'llleg':19, 'luleg':20, 'lfoot':21, 'rlleg':22, 'ruleg':23, 'rfoot':24}

        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()
        PART_PERMUTATION[2-x] = 3
        PART_PERMUTATION[3-x] = 2
        PART_PERMUTATION[4-x] = 5
        PART_PERMUTATION[5-x] = 4
        PART_PERMUTATION[6-x] = 7  # lebrow
        PART_PERMUTATION[7-x] = 6  # rebrow
        PART_PERMUTATION[13-x] = 16
        PART_PERMUTATION[14-x] = 17
        PART_PERMUTATION[15-x] = 18
        PART_PERMUTATION[16-x] = 13
        PART_PERMUTATION[17-x] = 14
        PART_PERMUTATION[18-x] = 15
        PART_PERMUTATION[19-x] = 22
        PART_PERMUTATION[20-x] = 23
        PART_PERMUTATION[21-x] = 24
        PART_PERMUTATION[22-x] = 19
        PART_PERMUTATION[23-x] = 20
        PART_PERMUTATION[24-x] = 21

    elif i==16:
    # [pottedplant]
        pimap = {'pot':1, 'plant':2}

        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()

    elif i==18:
    # [sofa]
        pimap = {}
    elif i==19:
    # [train]
        pimap = {'head':1, 'hfrontside':2, 'hleftside':3, 'hrightside':4, 'hbackside':5, 'hroofside':6}
        for ii in range(1,11):
            pimap['headlight_%d'%ii] = 10+ii
        for ii in range(1,11):
            pimap['coach_%d'%ii] = 20+ii
        for ii in range(1,11):
            pimap['cfrontside_%d'%ii] = 30+ii
        for ii in range(1,11):
            pimap['cleftside_%d'%ii] = 40+ii
        for ii in range(1,11):
            pimap['crightside_%d'%ii] = 50+ii
        for ii in range(1,11):
            pimap['cbackside_%d'%ii] = 60+ii
        for ii in range(1,11):
            pimap['croofside_%d'%ii] = 70+ii

        n_parts = max(pimap.values())
        PART_PERMUTATION = np.linspace(0, n_parts-1, n_parts, dtype=np.int32)+x
        PART_PERMUTATION = PART_PERMUTATION.tolist()
        PART_PERMUTATION[2-x] = 5  # hfrontside
        PART_PERMUTATION[5-x] = 2  # hbackside
        PART_PERMUTATION[3-x] = 4  # hleftside
        PART_PERMUTATION[4-x] = 3  # hrightside
        PART_PERMUTATION[41-x] = 51
        PART_PERMUTATION[51-x] = 41
        PART_PERMUTATION[42-x] = 52
        PART_PERMUTATION[52-x] = 42
        PART_PERMUTATION[43-x] = 53
        PART_PERMUTATION[53-x] = 43
        PART_PERMUTATION[44-x] = 54
        PART_PERMUTATION[54-x] = 44
        PART_PERMUTATION[45-x] = 55
        PART_PERMUTATION[55-x] = 45
        PART_PERMUTATION[46-x] = 56
        PART_PERMUTATION[56-x] = 46
        PART_PERMUTATION[47-x] = 57
        PART_PERMUTATION[57-x] = 47
        PART_PERMUTATION[48-x] = 58
        PART_PERMUTATION[58-x] = 48
        PART_PERMUTATION[49-x] = 59
        PART_PERMUTATION[59-x] = 49
        PART_PERMUTATION[50-x] = 60
        PART_PERMUTATION[60-x] = 50
        PART_PERMUTATION[31-x] = 61
        PART_PERMUTATION[61-x] = 31
        PART_PERMUTATION[32-x] = 62
        PART_PERMUTATION[62-x] = 32
        PART_PERMUTATION[33-x] = 63
        PART_PERMUTATION[63-x] = 33
        PART_PERMUTATION[34-x] = 64
        PART_PERMUTATION[64-x] = 34
        PART_PERMUTATION[35-x] = 65
        PART_PERMUTATION[65-x] = 35
        PART_PERMUTATION[36-x] = 66
        PART_PERMUTATION[66-x] = 36
        PART_PERMUTATION[37-x] = 67
        PART_PERMUTATION[67-x] = 37
        PART_PERMUTATION[38-x] = 68
        PART_PERMUTATION[68-x] = 38
        PART_PERMUTATION[39-x] = 69
        PART_PERMUTATION[69-x] = 39
        PART_PERMUTATION[40-x] = 70

    elif i==20:
    # [tvmonitor]
        pimap = {'screen':1}

            if PART_PERMUTATION is not None:
                # Adjust the permutation array to exclude the offset `x`
                PART_PERMUTATION_NEW = np.array(PART_PERMUTATION) - x
                n_parts_inclnone = len(PART_PERMUTATION_NEW)

                # Create a boolean array indicating parts with orientation, including entries for unassigned parts
                PART_WITH_ORIENTATION_inclnone = (np.linspace(0, n_parts_inclnone - 1, n_parts_inclnone, dtype=np.int32) - PART_PERMUTATION_NEW) != 0

                # Identify valid parts that are annotated in the dataset
                parts_inclnone_2_onlyvalid = [i in pimap.values() for i in range(1, n_parts_inclnone + 1)]

                # Filter the orientation array to include only valid parts
                PART_WITH_ORIENTATION = PART_WITH_ORIENTATION_inclnone[parts_inclnone_2_onlyvalid]

                def change_perm_sub(perm, flag_subdiv):
                    """
                    Adjust the permutation array to reflect only valid parts.

                    Args:
                        perm (numpy.ndarray): The original permutation array.
                        flag_subdiv (list): A boolean list indicating valid parts.

                    Returns:
                        numpy.ndarray: The adjusted permutation array for valid parts.
                    """
                    # Extract the subset of the permutation array corresponding to valid parts
                    perm_sub = perm[flag_subdiv]

                    # Get indices of the valid subset
                    subset_indices = np.nonzero(flag_subdiv)[0]

                    # Map global indices in the subset to local indices
                    index_map = {original: i for i, original in enumerate(subset_indices)}

                    # Map each entry in the subset permutation to its corresponding local index
                    perm_sub_mapped = np.array([index_map[i] for i in perm_sub])
                    return perm_sub_mapped

                # Update the permutation array to reflect valid parts only
                PART_PERMUTATION_NEW = change_perm_sub(PART_PERMUTATION_NEW, parts_inclnone_2_onlyvalid)

                # Convert the updated permutation array back to a list and reapply the offset `x`
                PART_PERMUTATION = (PART_PERMUTATION_NEW + x).tolist()

            else:
                # If no permutation is defined, set orientation to None
                PART_WITH_ORIENTATION = None
            

    return pimap, PART_PERMUTATION, PART_WITH_ORIENTATION