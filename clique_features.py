def clique_features(clique, edges, clique_idx, smile):
    NumAtoms = len(clique)  # 子结构中原子数
    NumEdges = 0  # 与子结构所连的边数，子结构的度
    for edge in edges:
        if clique_idx == edge[0] or clique_idx == edge[1]:
            NumEdges += 1
    mol = Chem.MolFromSmiles(smile)
    atoms = []
    NumHs = 0  # 基团中氢原子的个数
    NumImplicitValence = 0
    for idx in clique:
        atom = mol.GetAtomWithIdx(idx)
        atoms.append(atom.GetSymbol())
        NumHs += atom.GetTotalNumHs()
        NumImplicitValence += atom.GetImplicitValence()
    # 基团中是否包含环
    IsRing = 0
    if len(clique) > 2:
        IsRing = 1
    # 基团中是否有键
    IsBond = 0
    if len(clique) == 2:
        IsBond = 1
    return np.array(encoding_unk(atoms,
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding_unk(NumAtoms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding_unk(NumEdges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding_unk(NumHs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) +
                    one_hot_encoding_unk(NumImplicitValence, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) +
                    [IsRing] +
                    [IsBond])

def clique_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    clique, edge = cluster_graph(mol)

    c_features = []  # 特征矩阵
    for idx in range(len(clique)):
        cq_features = clique_features(clique[idx], edge, idx, smile)
        c_features.append(cq_features / sum(cq_features))

    clique_size = len(clique)  # 子结构图节点数
    return clique_size, c_features, edge
