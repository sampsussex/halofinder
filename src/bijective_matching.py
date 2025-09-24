import numpy as np
from collections import Counter
from typing import List
from numba import njit


@njit
def bijcheck(group_ids_1: List[int], group_ids_2: List[int], min_group_size: int):
    """
    Bijective comparison between two group catalogues as per Robotham+2011.
    This calculates Equations (9, 10, 12, and 13) from Robotham+2011.
    
    Args:
        group_ids_1: First group catalog (list of group IDs)
        group_ids_2: Second group catalog (list of group IDs)
        min_group_size: Minimum group size threshold
        
    Returns:
        BijResults containing e_num, e_den, q_num, q_den
    """
    assert len(group_ids_1) == len(group_ids_2), "Group catalogs must have same length"
    
    # Convert to numpy arrays for easier manipulation
    group_ids_1 = np.array(group_ids_1)
    group_ids_2 = np.array(group_ids_2)
    
    # Frequency tables excluding -1
    count_table_1 = Counter(group_ids_1[group_ids_1 != -1])
    count_table_2 = Counter(group_ids_2[group_ids_2 != -1])
    
    # Filter groups in tab1 with size >= min_group_size
    valid_groups_1 = [group for group, count in count_table_1.items() 
                     if count >= min_group_size]
    
    # Find indices of valid group members
    valid_mask = np.isin(group_ids_1, valid_groups_1)
    valid_indices_1 = np.where(valid_mask)[0]
    
    # Create group_list maintaining order (first occurrence of each group)
    group_list = []
    seen = set()
    for idx in valid_indices_1:
        group_id = group_ids_1[idx]
        if group_id not in seen:
            group_list.append(group_id)
            seen.add(group_id)
    
    # Process each group
    q1_values = []
    q2_values = []
    n1_values = []
    
    for group_id in group_list:
        # Find all galaxies in this group
        group_galaxies = np.where(group_ids_1 == group_id)[0]
        
        # Get corresponding groups in catalog 2
        overlap_groups = group_ids_2[group_galaxies]
        overlap_valid = overlap_groups[overlap_groups != -1]
        
        n1_current = count_table_1.get(group_id, 1)
        
        if len(overlap_valid) > 0:
            # Count overlaps
            temptab = Counter(overlap_valid)
            
            frac_1 = []
            frac_2 = []
            
            for group2, count in temptab.items():
                if group2 in count_table_2:
                    n2_val = count_table_2[group2]
                    frac_1.append(count / n1_current)
                    frac_2.append(count / n2_val)
            
            # Handle isolated galaxies (group_id = -1)
            num_isolated = np.sum(overlap_groups == -1)
            if num_isolated > 0:
                iso_frac1 = 1.0 / n1_current
                for _ in range(num_isolated):
                    frac_1.append(iso_frac1)
                    frac_2.append(1.0)
            
            # Find best match (first occurrence of maximum product)
            if frac_1:
                products = [f1 * f2 for f1, f2 in zip(frac_1, frac_2)]
                best_match = np.argmax(products)  # argmax returns first occurrence
                q1 = frac_1[best_match]
                q2 = frac_2[best_match]
            else:
                q1 = 1.0 / n1_current
                q2 = 1.0
        else:
            # All isolated
            q1 = 1.0 / n1_current
            q2 = 1.0
        
        q1_values.append(q1)
        q2_values.append(q2)
        n1_values.append(float(n1_current))
    
    # Calculate final results
    e_num = sum(1 for q1, q2 in zip(q1_values, q2_values) if q1 > 0.5 and q2 > 0.5)
    e_den = len(n1_values)
    q_num = sum(q1 * n1 for q1, n1 in zip(q1_values, n1_values))
    q_den = sum(n1_values)
    
    return e_num, e_den, q_num, q_den


@njit
def s_score(measured_groups: List[int], mock_groups: List[int], groupcut: int) -> float:
    """
    The final S-score measurement for comparisons between two group catalogues.
    Equation 15 of Robotham+2011.
    
    Args:
        measured_groups: Measured group catalog
        mock_groups: Mock/reference group catalog
        groupcut: Minimum group size threshold
        
    Returns:
        S-score value
    """
    e_num_mock, e_den_mock, q_num_mock, q_den_mock = bijcheck(mock_groups, measured_groups, groupcut)
    e_num_meas, e_den_meas, q_num_meas, q_den_meas = bijcheck(measured_groups, mock_groups, groupcut)
    
    mock_e = e_num_mock , e_den_mock
    fof_e = e_num_meas / e_den_meas
    mock_q = q_num_mock / q_den_mock
    fof_q = q_num_meas / q_den_meas
    
    return mock_e * fof_e * mock_q * fof_q, mock_e * fof_e, mock_q * fof_q