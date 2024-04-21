dataset_info = dict(
    dataset_name='marmoset',
    paper_info=dict(
        author='Chaoqun Cheng',
        title='MarmoPose: A Deep Learning-Based System for Multi-Marmoset 3D Real-Time Pose Tracking',
        container='bioRxiv',
        year='2024',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(name='head', 
             id=0, 
             color=[51, 153, 255], 
             type='upper', 
             swap=''),
        1:
        dict(
            name='leftear',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='rightear'),
        2:
        dict(
            name='rightear',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='leftear'),
        3:
        dict(
            name='neck',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        4:
        dict(
            name='leftelbow',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='rightelbow'),
        5:
        dict(
            name='rightelbow',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='leftelbow'),
        6:
        dict(
            name='lefthand',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='righthand'),
        7:
        dict(
            name='righthand',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='lefthand'),
        8:
        dict(
            name='spinemid',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        9:
        dict(
            name='tailbase',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        10:
        dict(
            name='leftknee',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='rightknee'),
        11:
        dict(
            name='rightknee',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='leftknee'),
        12:
        dict(
            name='leftfoot',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='rightfoot'),
        13:
        dict(
            name='rightfoot',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='leftfoot'),
        14:
        dict(
            name='tailmid',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap=''),
        15:
        dict(
            name='tailend',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('head', 'leftear'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('head', 'rightear'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('neck', 'leftear'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('neck', 'rightear'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('neck', 'leftelbow'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('neck', 'rightelbow'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('leftelbow', 'lefthand'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('rightelbow', 'righthand'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('neck', 'spinemid'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('spinemid', 'tailbase'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('tailbase', 'leftknee'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('tailbase', 'rightknee'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('leftknee', 'leftfoot'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('rightknee', 'rightfoot'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('tailbase', 'tailmid'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('tailmid', 'tailend'), id=15, color=[51, 153, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.034, 0.034, 0.034, 0.065, 0.107, 0.107, 0.083, 0.083, 0.058, 0.058,
        0.107, 0.107, 0.107, 0.107, 0.089, 0.089
    ])
