# start by importing some things we will need
import numpy as np
from math import floor, sqrt
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal
from dt_state_estimation.lane_filter.types import Segment, SegmentColor, SegmentPoint


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution around [0,0]
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0, cov_0)
    belief = RV.pdf(pos)
    return belief


def histogram_predict(belief, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
    belief_in = belief
    #  You will need  some parameters in the `robot_spec` defined above
    # You may find the following code useful to find the current best heading estimate:
    maxids = np.unravel_index(belief_in.argmax(), belief_in.shape)
    phi_max = grid_spec['phi_min'] + (maxids[1] + 0.5) * grid_spec['delta_phi']
    
 
    alpha = 2 * np.pi / robot_spec['encoder_resolution']
    rotation_wheel_left = alpha*left_encoder_ticks
    rotation_wheel_right = alpha*right_encoder_ticks
    d_left_wheel = rotation_wheel_left * robot_spec['wheel_radius']
    d_right_wheel = rotation_wheel_right * robot_spec['wheel_radius']

    v = (d_left_wheel + d_right_wheel) / 2
    w = (d_right_wheel - d_left_wheel) / robot_spec['wheel_baseline']

    # propagate each centroid
    d_t = grid_spec["d"] + v
    phi_t = grid_spec["phi"] + w

    p_belief = np.zeros(belief.shape)

    # Accumulate the mass for each cell as a result of the propagation step
    for i in range(belief.shape[0]):
        for j in range(belief.shape[1]):
            # If belief[i,j] there was no mass to move in the first place
            if belief[i, j] > 0:
                # Now check that the centroid of the cell wasn't propagated out of the allowable range
                if (
                    d_t[i, j] > grid_spec["d_max"]
                    or d_t[i, j] < grid_spec["d_min"]
                    or phi_t[i, j] < grid_spec["phi_min"]
                    or phi_t[i, j] > grid_spec["phi_max"]
                ):
                    continue
                
                diff_di = d_t[i, j] - grid_spec['d_min']
                diff_phi =  phi_t[i, j] - grid_spec['phi_min']
                i_new = int(diff_di / grid_spec["delta_d"]) 
                j_new = int(diff_phi / grid_spec["delta_phi"] )

                p_belief[i_new, j_new] += belief[i, j]

    # Finally we are going to add some "noise" according to the process model noise
    # This is implemented as a Gaussian blur
    s_belief = np.zeros(belief.shape)
    gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

    if np.sum(s_belief) == 0:
        return belief_in
    belief = s_belief / np.sum(s_belief)
    return belief


# We will start by doing a little bit of processing on the segments to remove anything that is
# behing the robot (why would it be behind?) or a color not equal to yellow or white


def prepare_segments(segments):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != SegmentColor.WHITE and segment.color != SegmentColor.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        filtered_segments.append(segment)
    return filtered_segments

def generate_vote(segment, road_spec):
    p1 = segment.points[0].as_array()
    p2 = segment.points[1].as_array()
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

    n_hat = np.array([-t_hat[1], t_hat[0]])
    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == SegmentColor.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec['linewidth_white']
        else:  # left edge of white lane

            d_i = -d_i

            phi_i = -phi_i
        d_i -= road_spec['lanewidth'] / 2

    elif segment.color == SegmentColor.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec['linewidth_yellow']
            phi_i = -phi_i
        else:  # right edge of white lane
            d_i = -d_i
        d_i = road_spec['lanewidth'] / 2 - d_i

    return d_i, phi_i



def generate_measurement_likelihood(segments, road_spec, grid_spec):
    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec["d"].shape)

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)

        # if the vote lands outside of the histogram discard it
        if (
            d_i > grid_spec["d_max"]
            or d_i < grid_spec["d_min"]
            or phi_i < grid_spec["phi_min"]
            or phi_i > grid_spec["phi_max"]
        ):
            continue

        # So now we have d_i and phi_i which correspond to the estimate of the distance
        # from the center and the angle relative to the center generated by the single
        # segment under consideration
        diff_di = d_i - grid_spec['d_min']
        diff_phi =  phi_i - grid_spec['phi_min']
        i = int(diff_di / grid_spec["delta_d"]) 
        j = int(diff_phi / grid_spec["delta_phi"] ) 

        # Add one vote to that cell
        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)

    if measurement_likelihood is not None:
        postbelief = (belief * measurement_likelihood)
        if np.sum(postbelief) == 0: 
            return measurement_likelihood, belief
        postbelief /= np.sum(postbelief)
    
    return measurement_likelihood, postbelief

def getSegmentDistance(segment):
    x_c = (segment.points[0].x + segment.points[1].x) / 2
    y_c = (segment.points[0].y + segment.points[1].y) / 2
    return sqrt(x_c**2 + y_c**2)