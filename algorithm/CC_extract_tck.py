# !/usr/bin/python
# -*- coding: utf-8 -*-

import nibabel.streamlines.array_sequence as nibAS


def extract_cc(imgtck):
    '''
    extract cc fiber
    :param streamlines:input wholeBrain fiber
    :return: ArraySequence: extract cc fiber
    '''
    L_temp = nibAS.ArraySequence()
    for i in range(len(imgtck.streamlines)):
        if imgtck.streamlines[i][0][0] * imgtck.streamlines[i][-1][0] < 0:
            L_temp.append(imgtck.streamlines[i])
    return L_temp


def extract_cc_step(imgtck):
    '''
    extract cc fiber
    :param streamlines:input wholeBrain fiber
    :return: ArraySequence: extract cc fiber
    '''
    L_temp = nibAS.ArraySequence()
    for i in range(len(imgtck.streamlines)):
        count = 0
        l_j = []
        if imgtck.streamlines[i][0][0] * imgtck.streamlines[i][-1][0] < 0:
            for j in range(len(imgtck.streamlines[i])-1):
                if imgtck.streamlines[i][j][0] * imgtck.streamlines[i][j+1][0] < 0:
                    count += 1
                    l_j.append(j)
                elif imgtck.streamlines[i][j][0] == 0:
                    count += 1
                    l_j.append(j)
            if count == 1 \
                    and (l_j[0] - 20) in range(len(imgtck.streamlines[i])) \
                    and (l_j[0] + 20) in range(len(imgtck.streamlines[i])) \
                    and imgtck.streamlines[i][l_j[0]-20][0] * imgtck.streamlines[i][l_j[0]+20][0] < 0:
                L_temp.append(imgtck.streamlines[i])
    return L_temp


if __name__ == '__main__':
    from rw.load import load_tck
    from rw.save import save_tck
    # load data
    file = '/home/brain/workingdir/data/dwi/hcp/preprocessed/' \
           'response_dhollander/100206/Diffusion/100k_sift_1M45006_dynamic250.tck'
    imgtck = load_tck(file)

    # extract CC
    L_temp = extract_cc_step(imgtck)
    # print L_temp

    # save data
    out_path = '/home/brain/workingdir/data/dwi/hcp/' \
               'preprocessed/response_dhollander/100206/result/CC_fib_new.tck'
    save_tck(L_temp, imgtck.header, imgtck.tractogram.data_per_streamline,
         imgtck.tractogram.data_per_point, imgtck.tractogram.affine_to_rasmm, out_path)
