def normalize_min_max(feat_matrix):
    return (feat_matrix - feat_matrix.min()) / (feat_matrix.max() - feat_matrix.min())
