from numba import jit


@jit(nopython=True)
def apply_creation(create_site, x_prime):
    multiplicator = 1-x_prime[create_site]

    if multiplicator != 0:
        x_prime[create_site] = 1

    return multiplicator

@jit(nopython=True)
def apply_annihilation(annihilate_site, x_prime):
    multiplicator = x_prime[annihilate_site]

    if multiplicator != 0:
        x_prime[annihilate_site] = 0

    return multiplicator

@jit(nopython=True)
def apply_hopping(annihilate_site, create_site, x_prime):
    start_occ = x_prime[annihilate_site]
    final_occ = x_prime[create_site]

    if start_occ != final_occ and start_occ == 1:
        multiplicator = 1
    else:
        multiplicator = 0

    if multiplicator != 0:
        x_prime[annihilate_site] = final_occ
        x_prime[create_site] = start_occ

    return multiplicator


@jit(nopython=True)
def apply_particle_hole(particle_site, hole_site, x_prime):
    start_occ = x_prime[particle_site]
    final_occ = x_prime[hole_site]

    if start_occ != final_occ and start_occ == 1:
        multiplicator = 1
    else:
        multiplicator = 0

    return multiplicator
