from lsat.typing import KeypointData


def group_kds(kds: list[KeypointData]) -> list[list[KeypointData]]:
    'Groups keypoint data objects that belong to same frame'
    grouped: list[list[KeypointData]] = [[]]
    for kd in kds:
        added = False
        for g in grouped:
            if not added and (not g or g[-1]['image_id'] != kd['image_id']):
                g.append(kd)
                added = True
        if not added:
            grouped.append([kd])
    return grouped
