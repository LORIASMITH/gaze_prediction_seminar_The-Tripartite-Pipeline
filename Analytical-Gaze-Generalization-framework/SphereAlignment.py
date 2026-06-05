import DataProcessFuncs as funcs
import numpy as np
import time
from cv2 import Rodrigues
import sklearn.manifold
from scipy.optimize import least_squares


def InferenceGPM(p, pts, debug=False):
    pts = pts - p[10:13]
    pts = pts / np.linalg.norm(pts, ord=2, axis=1, keepdims=True)
    R, _ = Rodrigues(p[:3])
    pts_rotated = (R @ pts.reshape((pts.shape[0], 3, 1))).reshape(pts.shape)
    yaw = np.arctan2(-pts_rotated[:, 0], -pts_rotated[:, 2])
    yaw = p[6] * yaw + p[7]
    pitch = np.arcsin(-pts_rotated[:, 1])
    pitch = p[8] * pitch + p[9]
    return funcs.gazeTo3d_array(np.hstack((yaw.reshape(len(yaw), 1), pitch.reshape(len(pitch), 1))))


def SolveGPM(p, pts, label):
    return funcs.angular_batch(InferenceGPM(p,  pts), label)


class GeodesicProjection():
    def __init__(self, pts, label, params=None):
        if params is None:
            self.label = funcs.gazeTo3d_array(label)
            #          0      1     2     3     4    5     6  7   8   9    10   11   12
            #         R0_x, R0_y, R0_z, R1_x, R1_y, R1_z, k0, b0, k1, b1, O_x, O_y, O_z
            self.p0 = [0.1, 0.1,  0.1,  0.1,   0.1, 0.1,   1,  0, 1,   0,   0,   0,   4.48]
            self.pts = pts

            ls_result = least_squares(SolveGPM, self.p0, args=(self.pts, self.label),
                                   bounds=([-4, -4, -4, -4, -4, -4,  -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf],
                                           [4, 4, 4, 4, 4, 4, np.inf, np.pi, np.inf, np.pi, np.inf, np.inf, np.inf]))
                                # bounds=([-4, -4, -4, -4, -4, -4,  -1.5, -5/180*np.pi, -1.5, -5/180*np.pi, -np.inf, -np.inf, -np.inf],
                                #            [4, 4, 4, 4, 4, 4, 1.5, 5/180*np.pi, 1.5, 5/180*np.pi, np.inf, np.inf, np.inf]))
            self.result = ls_result.x
            origin = self.result[10:13]
            Rs = np.sqrt(np.sum((pts - origin) ** 2, axis=1))

            self.result = np.hstack((self.result, np.mean(Rs)))
        else:
            self.pts = pts
            self.result = params
            origin = self.result[10:13]
            Rs = np.sqrt(np.sum((pts - origin) ** 2, axis=1))

        # print(f'GeodesicProjection: Sphere Error - {np.mean(abs(Rs - self.result[-1])/self.result[-1] * 100):.2f}%, {np.mean(np.sqrt(Rs))}')


    def GetResult(self):
        return self.result, None, None

    def __call__(self, pts):
        return InferenceGPM(self.result,  pts, debug=False)


def ISOMap(data, dim=3, n_neighbors=300, fitter=None, verbose=False):
    begin = time.time()
    if n_neighbors is None:
        n_neighbors = int(data.shape[0]*0.4)
    if fitter is None:
        if verbose:
            print(f"[State info ] Start New Isomap ...")
        ISO_fitter = sklearn.manifold.Isomap(n_neighbors=n_neighbors, n_components=dim)
        fitter = ISO_fitter.fit(data)
    else:
        if verbose:
            print(f"[State info ] ISOMap from Existing Param...")
    PGF = fitter.transform(data)
    if verbose:
        print(f"[State info ] Successful! ISOMap complete in {(time.time() - begin):.2f}s")
    return PGF, fitter


def FitGaze(PGF, gaze_label, param=None, verbose=True):
    if param is None:
        if verbose:
            print(f'[State info ] Start a new Geodesic Projection....')

        model = GeodesicProjection(PGF, gaze_label, param)
    else:
        if verbose:
            print(f'[State info ] Geodesic Projection from Existing Parameter!')
        model = GeodesicProjection(PGF, None, param)
    fitting_param, _, _ = model.GetResult()
    sphere = fitting_param[10:]
    preds = model(PGF)
    if verbose:
        print(
            f'[State info ] Fitting Error: {np.mean(funcs.angular_batch(preds, gaze_label)) * 180 / np.pi}')
        print(f'sphere: {sphere}')
    return fitting_param, preds, model