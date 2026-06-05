# Run complete training and testing process for AGG

# -------Pretrain baseline model on source domain-------
python Baseline.py --phase=train,test --source=ETH --target=ETH
python Baseline.py --phase=test --source=ETH --target=MPII
python Baseline.py --phase=test --source=ETH --target=EyeDiapAll

python Baseline.py --phase=train,test --source=Gaze360 --target=Gaze360
python Baseline.py --phase=test --source=Gaze360 --target=MPII
python Baseline.py --phase=test --source=Gaze360 --target=EyeDiapAll


# -------Geodesic Projection Module-------
python GeodesicProjectionModule.py --phase=train,test --source=ETH --target=MPII
python GeodesicProjectionModule.py --phase=test --source=ETH --target=EyeDiapAll

python GeodesicProjectionModule.py --phase=train,test --source=Gaze360 --target=MPII
python GeodesicProjectionModule.py --phase=test --source=Gaze360 --target=EyeDiapAll

python GeodesicProjectionModule.py --phase=test --source=ETH --target=ETH
python GeodesicProjectionModule.py --phase=test --source=Gaze360 --target=Gaze360


# -------Isometric Propagator-------
python IP.py --phase=train,test --source=ETH --target=MPII
python IP.py --phase=test --source=ETH --target=EyeDiapAll

python IP.py --phase=train,test --source=Gaze360 --target=MPII
python IP.py --phase=test --source=Gaze360 --target=EyeDiapAll

python IP.py --phase=test --source=ETH --target=ETH
python IP.py --phase=test --source=Gaze360 --target=Gaze360


# -------Sphere Oriented Training-------
python SphereOrientedTraining.py --phase=train,test --source=ETH --target=ETH
python SphereOrientedTraining.py --phase=test --source=ETH --target=MPII
python SphereOrientedTraining.py --phase=test --source=ETH --target=EyeDiapAll

python SphereOrientedTraining.py --phase=train,test --source=Gaze360 --target=Gaze360
python SphereOrientedTraining.py --phase=test --source=Gaze360 --target=MPII
python SphereOrientedTraining.py --phase=test --source=Gaze360 --target=EyeDiapAll


# -------Test model after SOT -------
python TestSOT.py --phase=test --source=ETH --target=ETH
python TestSOT.py --phase=test --source=ETH --target=MPII
python TestSOT.py --phase=test --source=ETH --target=EyeDiapAll

python TestSOT.py --phase=test --source=Gaze360 --target=Gaze360
python TestSOT.py --phase=test --source=Gaze360 --target=MPII
python TestSOT.py --phase=test --source=Gaze360 --target=EyeDiapAll