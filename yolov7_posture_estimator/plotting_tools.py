import typing as tp
import numpy as np
import matplotlib.pyplot as plt


def get_chain_dots(
        dots: np.ndarray,   # shape == (n_dots, 3)
        chain_dots_indexes: tp.List[int], # length == n_dots_in_chain
                                          # in continuous order, i.e. 
                                          # left_hand_ix >>> chest_ix >>> right_hand_ix
        ) -> np.ndarray:    # chain of dots
    """Get continuous chain of dots
    
    chain_dots_indexes - 
        indexes of points forming a continuous chain;
        example of chain: [hand_l, elbow_l, shoulder_l, chest, shoulder_r, elbow_r, hand_r]
    """
    return dots[chain_dots_indexes]


def get_chains(
        dots: np.ndarray,   # shape == (n_dots, 3)
        spine_chain_ixs: tp.List[int], # pelvis >>> chest >>> head
        hands_chain_ixs: tp.List[int], # left_hand >>> chest >>> right_hand
        legs_chain_ixs: tp.List[int]   # left_leg >>> pelvis >>> right_leg
        ):
    return (get_chain_dots(dots, spine_chain_ixs),
            get_chain_dots(dots, hands_chain_ixs),
            get_chain_dots(dots, legs_chain_ixs))


def subplot_nodes(dots: np.ndarray, # shape == (n_dots, 3)
                  ax):
    return ax.scatter3D(*dots.T, c=dots[:, -1])


def subplot_bones(chains: tp.Tuple[np.ndarray, ...], ax):
    return [ax.plot(*chain.T) for chain in chains]


def plot_skeletons(
        skeletons: tp.Sequence[np.ndarray], 
        chains_ixs: tp.Tuple[tp.List[int], tp.List[int], tp.List[int]]):
    fig = plt.figure()
    for i, dots in enumerate(skeletons, start=1):
        chains = get_chains(dots, *chains_ixs)
        ax = fig.add_subplot(2, 5, i, projection='3d')
        subplot_nodes(dots, ax)
        subplot_bones(chains, ax)
    plt.show()


def test():
    """Plot random poses of simplest skeleton"""
    skeletons = np.random.standard_normal(size=(10, 11, 3))
    chains_ixs = ([0, 1, 2, 3, 4],  # hand_l, elbow_l, chest, elbow_r, hand_r
                  [5, 2, 6],        # pelvis, chest, head
                  [7, 8, 5, 9, 10]) # foot_l, knee_l, pelvis, knee_r, foot_r
    plot_skeletons(skeletons, chains_ixs)


if __name__ == '__main__':
    test()
