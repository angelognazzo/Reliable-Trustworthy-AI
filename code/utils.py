import torch

def tight_bounds(lower_bound, upper_bound, lower_bound_tmp, upper_bound_tmp):
    # get the tightest bounds possible
    # lower_bound and lower_bound_tmp could or couldn't have an interesection
    # EASY CASE: there is an interesection between the two bounds:
    # check the intersection condition
    mask_positive = torch.max(lower_bound_tmp, lower_bound) <= torch.min(
        upper_bound_tmp, upper_bound)
    mask_negative = torch.logical_not(mask_positive)
    # if there is an intersection, the bounds can be tighten by taking the greatest maximum and the smallest minimum
    lower_bound = torch.where(mask_positive, torch.max(
        lower_bound_tmp, lower_bound), lower_bound)
    upper_bound = torch.where(mask_positive, torch.min(
        upper_bound_tmp, upper_bound), upper_bound)

    assert (lower_bound <= upper_bound).all(
    ), "DeepPolyNetwork forward: Error with the box bounds: lower > upper"

    # there is no intersection between the two bounds, bounds are therefore 'disjoint'
    # check if the new bounds are tighter than the old ones and if upper_bound_tmp > lower_bound_tmp.
    # This (upper_bound_tmp > lower_bound_tmp) could happen during backsubstitution where upper_bound_tmp
    # and lower_bound_tmp are computed using two different calls to the 'swap_and_forward' function
    # and therefore it's possible that in some cases the correct order is not preserved
    mask_tighter_disjoint = (upper_bound_tmp - lower_bound_tmp < upper_bound -
                             lower_bound) & (upper_bound_tmp - lower_bound_tmp >= 0)
    # use also mask_negative to update the entries that haven't been updated above
    lower_bound = torch.where(
        mask_negative & mask_tighter_disjoint, lower_bound_tmp, lower_bound)
    upper_bound = torch.where(
        mask_negative & mask_tighter_disjoint, upper_bound_tmp, upper_bound)
    
    # the correct order now should be respected
    assert (lower_bound <= upper_bound).all(
    ), "DeepPolyNetwork forward: Error with the box bounds: lower > upper"

    return lower_bound, upper_bound
