#!/usr/bin/python3

from pathlib import Path
import copy
import numpy as np
import random
import os
import json
import pandas as pd
import cv2
import argparse
import sys
import tqdm

det_filename = 'bboxes.json'
gt_path = 'bdd100k/labels'
save_path = './'

def precision(TP, FP):
    return TP/(TP+FP)

def recall(TP, FN):
    return TP/(TP+FN)

def f1_score(TP, FP, FN):
    return TP/(TP+0.5*(FP+FN))

def associator(gt_bboxes, pred_bboxes, conf, type:str):
    # print(f"Pred size = {len(pred_bboxes)}  |  Gt size = {len(gt_bboxes)}")
    assert len(pred_bboxes) == len(conf), f"The quantity of boxes doesn't match the number of confidence values"
    # assert len(conf) != 0, f'Cannot associate boxes: Empty prediction list'
    # Reorganize by confidence value
    if len(conf) == 0:
        associated = []
        for box in gt_bboxes:
            associated.append([box, None, 0])
        return associated
    conf, pred_bboxes = zip(*reversed(sorted(zip(conf, pred_bboxes))))
    conf = list(conf)
    pred_bboxes = list(pred_bboxes)
    
    iou_table = []
    # Rows --> Ground Truth
    for gt in gt_bboxes:
        line = []
        # Columns --> Predictions
        for pred in pred_bboxes:
            iou = IoU_calc(gt ,pred, type)
            line.append(iou)
        iou_table.append(line)
    iou_table = np.asarray(iou_table)
    maxim = np.max(iou_table)
    associated = []
    while maxim > 0 and iou_table.shape[1]>0:
        row_idx, column_idx = np.unravel_index(np.argmax(iou_table), iou_table.shape)
        # Caso o máximo seja único na linha e na coluna
        if np.count_nonzero(iou_table[row_idx, :] == maxim) == 1 and np.count_nonzero(iou_table[:, column_idx] == maxim) == 1:
            associated.append([gt_bboxes[row_idx], pred_bboxes[column_idx], iou_table[row_idx, column_idx]])
            iou_table = np.delete(iou_table, obj=row_idx, axis=0) # axis=0 --> linhas
            iou_table = np.delete(iou_table, obj=column_idx, axis=1)
            del gt_bboxes[row_idx]
            del pred_bboxes[column_idx]
            del conf[column_idx]
        else:
            # Obter posições com valor máximo na mesma linha e coluna
            max_idx = np.argwhere(iou_table == maxim)
            idx_remove = []
            for idx, [row, column] in enumerate(max_idx):
                if (row != row_idx and column != column_idx):
                    idx_remove.append(idx)
            max_idx = np.delete(max_idx, obj=idx_remove, axis=0)
            # Remover os que têm menor conf
            confs = [conf[confi] for confi in max_idx[:,1]]
            confi = 0
            idx_keep = []
            first_run = 1
            for idx, confd in enumerate(confs):
                if confd > confi:
                    confi = confd
                    idx_keep = [idx]
                elif confd == confi:
                    idx_keep.append(idx)
            max_idx = max_idx[idx_keep]
            # Remover de forma aleatória
            if max_idx.shape[0] > 1:
                max_idx = max_idx[random.randint(0,max_idx.shape[0]-1)]
            max_idx = np.squeeze(max_idx)
            associated.append([gt_bboxes[max_idx[0]], pred_bboxes[max_idx[1]], iou_table[max_idx[0], max_idx[1]]])
            iou_table = np.delete(iou_table, obj=max_idx[0], axis=0)
            iou_table = np.delete(iou_table, obj=max_idx[1], axis=1)
            del gt_bboxes[max_idx[0]]
            del pred_bboxes[max_idx[1]]
            del conf[max_idx[1]]
        if iou_table.shape[1]<=0 or iou_table.shape[0] == 0:
            break
        maxim = np.max(iou_table)
    for box in gt_bboxes:
        associated.append([box, None, 0])
    for box in pred_bboxes:
        associated.append([None, box, 0])
    return associated

def IoU_mask(maskA, maskB):
    intersection = np.logical_and(maskA, maskB)
    union = np.logical_or(maskA, maskB)
    sum_union = np.sum(union)
    if sum_union != 0:
        iou = np.sum(intersection) / sum_union
    else:
        iou = 0
    return iou
    
def IoU_bbox(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute IoU
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
    
def IoU_calc(gt ,pred, type:str):
    if type == "box":
        iou = IoU_bbox(gt, pred)
        return iou
    elif type == "mask":
        iou = IoU_mask(gt, pred)
        return iou
    else:
        raise Exception("Invalid IoU type")
    
def AP_aux(iou, iou_tresh):
    # return TP, FP, FN
    if iou >= iou_tresh: # False positive
        return 1, 0, 0
    elif iou < iou_tresh and iou > 0: # False positive and false negative
        return 0, 1, 1
    else:
        raise Exception("Invalid IoU")
        
def Evaluation_TFPN(associated_boxes, iou_tresh):
    TP, FP, FN = 0, 0, 0
    for element in associated_boxes:
        gt_bbox, pred_bbox, iou = element
        if gt_bbox is None:
            FP += 1
        elif pred_bbox is None:
            FN += 1
        else:
            TPn, FPn, FNn = AP_aux(iou, iou_tresh)
            TP, FP, FN = TP + TPn, FP + FPn, FN + FNn
    return TP, FP, FN

def precision_recall(gt_bboxes, pred_bboxes, conf, iou_tresh, type):
    gt_bboxes_send = copy.deepcopy(gt_bboxes)
    pred_bboxes_send = copy.deepcopy(pred_bboxes)
    conf_send = copy.deepcopy(conf)
    # TP, FP, FN = Evaluation_TFPN(box_association(gt_bboxes_send, pred_bboxes_send, conf_send),iou_tresh)
    TP, FP, FN = Evaluation_TFPN(associator(gt_bboxes_send, pred_bboxes_send, conf_send, type),iou_tresh)
    return precision(TP, FP), recall(TP, FN)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                            prog = 'eval_postprocess',
                            description='This node send an image and receives and\
                                saves the inference results into a folder')
    
    parser.add_argument('-fn', '--folder_name', type=str, 
                        dest='folder_name', required=True, 
                        help='Name of the folder with results')

    parser.add_argument('-nimg', '--number_images', type=int, 
                        dest='number_images', required=True, 
                        help='Quantity of images for evaluation')
    
    parser.add_argument('-obj', '--obj_dect', type=int, 
                        dest='obj_dect', required=True, 
                        help='True if object detection evaluation')
    
    parser.add_argument('-l', '--lane', type=int, 
                        dest='lane', required=True, 
                        help='True if lane detection evaluation')
    
    parser.add_argument('-da', '--drivable', type=int, 
                        dest='drivable', required=True, 
                        help='True if drivable area evaluation')
    
    parser.add_argument('-out', '--output_name', type=str, 
                        dest='output_name', required=False,
                        default= None,
                        help='The name of the xls file')
    
    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))
    
    n_images = args['number_images']
    preds_path = args['folder_name'] + '/labels/'
    mode_obj_dect = args['obj_dect']
    mode_drivable = args['drivable']
    mode_lane = args['lane']
    file_name = args['output_name'] if not(args['output_name'] is None) else args['folder_name']
    curr_path = Path(__file__).parent
    preds_path = str(curr_path / preds_path) + '/'
    # temp
    f2 = open(str(curr_path / gt_path/'det_20/det_val.json'))
    gt_detections = json.load(f2)

    if mode_obj_dect:
        det_path = str(curr_path / preds_path /'det_20'/det_filename)
        f = open(str(det_path))
        f2 = open(str(curr_path / gt_path/'det_20/det_val.json'))
        detections = json.load(f)
        print('Joining truck, bus and car classes...')
        for image in detections:
            if 'car' in detections[image]['Bboxes']:
                car = detections[image]['Bboxes']['car']
            else:
                car = []
            if 'truck' in detections[image]['Bboxes']:
                # print(detections[image]['Bboxes']['truck'])
                car.extend(detections[image]['Bboxes']['truck'])
            if 'bus' in detections[image]['Bboxes']:
                car.extend(detections[image]['Bboxes']['bus'])
            if len(car)>0:
                detections[image]['Bboxes']['car'] = car
        gt_detections = json.load(f2)

    if mode_drivable:
        drivable_path = str(curr_path / preds_path /'drivable/masks')
        f3 = open(drivable_path + '/../drivable.json')
        drivable = json.load(f3)
        gt_drivable_list = sorted(os.listdir(curr_path / gt_path/'drivable/masks'))
        drivable_list = sorted(os.listdir(drivable_path))

    if mode_lane:
        lane_path = str(curr_path / preds_path /'lane/masks')
        f4 = open(lane_path + '/../lane.json')
        lane = json.load(f4)
        gt_lane_list = sorted(os.listdir(curr_path / gt_path/'lane/masks'))
        lane_list = sorted(os.listdir(lane_path))

    if mode_obj_dect and mode_drivable and mode_lane:
        columns_Resume = pd.MultiIndex.from_tuples([('Object detection','Precision@50'),('Object detection','Recall@50'),
                                                    ('Object detection','Precision@75'),('Object detection','Recall@75'),
                                                    ('Object detection','Ti'),('Object detection','Tf'),
                                                    ('Drivable area','IoU'),('Drivable area','Ti'),('Drivable area','Tf'),
                                                    ('Lane marking','IoU'),('Lane marking','Ti'),('Lane marking','Tf')
                                                    ])
    elif mode_drivable and mode_lane:
        columns_Resume = pd.MultiIndex.from_tuples([('Drivable area','IoU'),('Drivable area','Ti'),('Drivable area','Tf'),
                                                    ('Lane marking','IoU'),('Lane marking','Ti'),('Lane marking','Tf')
                                                    ])
    elif mode_obj_dect:
        columns_Resume = pd.MultiIndex.from_tuples([('Object detection','Precision@50'),('Object detection','Recall@50'),
                                                    ('Object detection','Precision@75'),('Object detection','Recall@75'),
                                                    ('Object detection','Ti'),('Object detection','Tf')
                                                    ])
    elif mode_drivable:
        columns_Resume = pd.MultiIndex.from_tuples([('Drivable area','IoU'),('Drivable area','Ti'),('Drivable area','Tf')
                                                    ])
    elif mode_lane:
        columns_Resume = pd.MultiIndex.from_tuples([('Lane marking','IoU'),('Lane marking','Ti'),('Lane marking','Tf')
                                                    ])
    else:
        print('Edit the code and configure other mode manualy! Exiting...')
        exit()



    a = pd.MultiIndex.from_tuples([('a','a')])
    df = pd.DataFrame(columns=columns_Resume, index=a)
    df = df.drop(['a'])
    for image_idx in tqdm.tqdm(range(n_images)):
        # print(image_idx)
        # arranjar outra forma
        image_name = gt_detections[image_idx]['name'].split('.')[0] + '.jpg'
        try:
            if mode_obj_dect:
                # Object detection
                boxes = {}
                for idx in range(len(gt_detections[image_idx]['labels'])):
                    if gt_detections[image_idx]['labels'][idx]['category'] == 'truck' or gt_detections[image_idx]['labels'][idx]['category'] == 'bus': # Some networks join this classes into car
                        gt_detections[image_idx]['labels'][idx]['category'] = 'car'
                    
                    if gt_detections[image_idx]['labels'][idx]['category'] == 'car': # Some networks only detect cars; this is to avoid unnecessary computing
                        box = [gt_detections[image_idx]['labels'][idx]['box2d']['x1'], gt_detections[image_idx]['labels'][idx]['box2d']['y1'], gt_detections[image_idx]['labels'][idx]['box2d']['x2'], gt_detections[image_idx]['labels'][idx]['box2d']['y2']]
                        if gt_detections[image_idx]['labels'][idx]['category'] in boxes:
                            boxes[gt_detections[image_idx]['labels'][idx]['category']].append(box)
                        else:
                            boxes[gt_detections[image_idx]['labels'][idx]['category']] = [box]

                if image_name in detections:
                    # for classe in detections[image_name]['Bboxes']:
                    # print(boxes)
                    for classe in boxes:
                        gt_list = boxes[classe]
                        if classe in detections[image_name]['Bboxes']:
                            pred_list = detections[image_name]['Bboxes'][classe]
                            # TODO add conf list
                            conf_list = []
                            conf_list.extend([1] * len(pred_list))
                            pr_50 = precision_recall(gt_list, pred_list, conf_list, 0.5, "box")
                            df.loc[(image_name,classe),('Object detection','Precision@50')] = pr_50[0]
                            df.loc[(image_name,classe),('Object detection','Recall@50')] = pr_50[1]
                            pr_75 = precision_recall(gt_list, pred_list, conf_list, 0.75, "box")
                            df.loc[(image_name,classe),('Object detection','Precision@75')] = pr_75[0]
                            df.loc[(image_name,classe),('Object detection','Recall@75')] = pr_75[1]
                            df.loc[(image_name,classe),('Object detection','Ti')] = detections[image_name]['Start']
                            df.loc[(image_name,classe),('Object detection','Tf')] = detections[image_name]['End']
                        # else:
                        #     pred_list = []
                        #     conf_list = []
                        # pr_50 = precision_recall(gt_list, pred_list, conf_list, 0.5, "box")
                        # df.loc[(image_name,classe),('Object detection','Precision@50')] = pr_50[0]
                        # df.loc[(image_name,classe),('Object detection','Recall@50')] = pr_50[1]
                        # pr_75 = precision_recall(gt_list, pred_list, conf_list, 0.75, "box")
                        # df.loc[(image_name,classe),('Object detection','Precision@75')] = pr_75[0]
                        # df.loc[(image_name,classe),('Object detection','Recall@75')] = pr_75[1]
                        # df.loc[(image_name,classe),('Object detection','Ti')] = detections[image_name]['Start']
                        # df.loc[(image_name,classe),('Object detection','Tf')] = detections[image_name]['End']
            new_name = image_name.split('.')[0] + '.png'
            if mode_drivable:
                # Drivable area
                # new_name = image_name.split('.')[0] + '.png'
                gt_mask = cv2.imread(str(curr_path / gt_path/'drivable/masks/val'/ new_name))
                gt_mask = (gt_mask != np.max(gt_mask)) * 255
                gt_mask = gt_mask.astype(np.uint8)
                pred_mask = cv2.imread(drivable_path + '/' + new_name)
                iou = IoU_mask(gt_mask, pred_mask)
                df.loc[(image_name,'Drivable'),('Drivable area','IoU')] = iou
                df.loc[(image_name,'Drivable'),('Drivable area','Ti')] = drivable[image_name]['Start']
                df.loc[(image_name,'Drivable'),('Drivable area','Tf')] = drivable[image_name]['End']
            if mode_lane:
                # Lane Marking
                # gt_mask = cv2.imread(str(curr_path / gt_path/'lane/colormaps/val'/ new_name))
                gt_mask = cv2.imread(str(curr_path / gt_path/'lane/transformed_masks/val'/ new_name))
                # gt_mask = (gt_mask != 0) * 255
                # gt_mask = gt_mask.astype(np.uint8)
                pred_mask = cv2.imread(lane_path + '/' + new_name)
                iou = IoU_mask(gt_mask, pred_mask)
                df.loc[(image_name,'Lane'),('Lane marking','IoU')] = iou
                df.loc[(image_name,'Lane'),('Lane marking','Ti')] = lane[image_name]['Start']
                df.loc[(image_name,'Lane'),('Lane marking','Tf')] = lane[image_name]['End']

        except:
            print('Erro!')
    print('Reorganizing and saving...')
    classes_list = [*set([df.index[x][1][:] for x in range(len(df.index))])]

    if mode_lane:
        classes_list.remove('Lane')

    if mode_drivable:
        classes_list.remove('Drivable')
    
    if mode_obj_dect and mode_drivable and mode_lane:
        columns_Resume = pd.MultiIndex.from_tuples([('Object detection','AP@50'),('Object detection','AP@75'), ('Object detection','FPS'),#('Object detection','Ti'),('Object detection','Tf'),
                                                    ('Drivable area','IoU'), ('Drivable area','FPS'),#('Drivable area','Ti'),('Drivable area','Tf'),
                                                    ('Lane marking','IoU'), ('Lane marking','FPS')#,('Lane marking','Ti'),('Lane marking','Tf')
                                                    ])
    elif mode_drivable and mode_lane:
        columns_Resume = pd.MultiIndex.from_tuples([('Drivable area','IoU'), ('Drivable area','FPS'),#('Drivable area','Ti'),('Drivable area','Tf'),
                                                    ('Lane marking','IoU'), ('Lane marking','FPS')#,('Lane marking','Ti'),('Lane marking','Tf')
                                                    ])
    elif mode_obj_dect:
        columns_Resume = pd.MultiIndex.from_tuples([('Object detection','AP@50'),('Object detection','AP@75'), ('Object detection','FPS')#('Object detection','Ti'),('Object detection','Tf'),
                                                    ])
    elif mode_drivable:
        columns_Resume = pd.MultiIndex.from_tuples([('Drivable area','IoU'), ('Drivable area','FPS')#('Drivable area','Ti'),('Drivable area','Tf'),
                                                    ])
    elif mode_lane:
        columns_Resume = pd.MultiIndex.from_tuples([('Lane marking','IoU'), ('Lane marking','FPS')#,('Lane marking','Ti'),('Lane marking','Tf')
                                                    ])
    else:
        print('Edit the code and configure other mode manualy! Exiting...')
        exit()


    df2 = pd.DataFrame(columns=columns_Resume)

    if mode_obj_dect:
        for classe in classes_list:
            for tresh in ['50', '75']:
                # Get precision and recall values
                p_50 = df.loc[(slice(None), classe),('Object detection','Precision@'+tresh)].tolist()
                r_50 = df.loc[(slice(None), classe),('Object detection','Recall@'+tresh)].tolist()
                # Organize by recall order
                r_50, p_50 = zip(*sorted(zip(r_50, p_50)))
                # Get average precision
                ap = 0
                for idx in range(len(p_50)):
                    if idx>=1:
                        ap += (r_50[idx]-r_50[idx-1])*p_50[idx]
                df2.loc[(classe),('Object detection','AP@'+tresh)] = ap
        fps = np.average(1/np.subtract(np.array(df.loc[(slice(None), 'car'), ('Object detection','Tf')].tolist()), np.array(df.loc[(slice(None), 'car'),('Object detection','Ti')].tolist())))
        df2.loc[(classe),('Object detection','FPS')] = fps

    if mode_lane:
        lane_iou = np.average(df.loc[(slice(None), 'Lane'),('Lane marking','IoU')].tolist())
        fps = np.average(1/np.subtract(np.array(df.loc[(slice(None), 'Lane'),('Lane marking','Tf')].tolist()), np.array(df.loc[(slice(None), 'Lane'),('Lane marking','Ti')].tolist())))
        df2.loc[('Lane'),('Lane marking','IoU')] = lane_iou
        df2.loc[('Lane'),('Lane marking','FPS')] = fps
    if mode_drivable:
        drivable_iou = np.average(df.loc[(slice(None), 'Drivable'),('Drivable area','IoU')].tolist())
        fps = np.average(1/np.subtract(np.array(df.loc[(slice(None), 'Drivable'),('Drivable area','Tf')].tolist()), np.array(df.loc[(slice(None), 'Drivable'),('Drivable area','Ti')].tolist())))
        print(fps)
        df2.loc[('Drivable'),('Drivable area','IoU')] = drivable_iou
        df2.loc[('Drivable'),('Drivable area','FPS')] = fps

    # columns_Resume = pd.MultiIndex.from_tuples([('Object detection','mAP@50'),('Object detection','mAP@75'),#('Object detection','Ti'),('Object detection','Tf'),
    #                                             ('Drivable area','mIoU'),#('Drivable area','Ti'),('Drivable area','Tf'),
    #                                             ('Lane marking','mIoU')#,('Lane marking','Ti'),('Lane marking','Tf')
    #                                             ])



    # if mode_obj_dect and mode_drivable and mode_lane:
    #     columns_Resume = pd.MultiIndex.from_tuples([('Object detection','mAP@50'),('Object detection','mAP@75'),#('Object detection','Ti'),('Object detection','Tf'),
    #                                                 ('Drivable area','mIoU'),#('Drivable area','Ti'),('Drivable area','Tf'),
    #                                                 ('Lane marking','mIoU')#,('Lane marking','Ti'),('Lane marking','Tf')
    #                                                 ])
    # elif mode_drivable and mode_lane:
    #     columns_Resume = pd.MultiIndex.from_tuples([('Drivable area','mIoU'),#('Drivable area','Ti'),('Drivable area','Tf'),
    #                                                 ('Lane marking','mIoU')#,('Lane marking','Ti'),('Lane marking','Tf')
    #                                                 ])
    # elif mode_obj_dect:
    #     columns_Resume = pd.MultiIndex.from_tuples([('Object detection','mAP@50'),('Object detection','mAP@75'), ('Object detection','FPS')#('Object detection','Ti'),('Object detection','Tf'),
    #                                                 ])
    # elif mode_drivable:
    #     columns_Resume = pd.MultiIndex.from_tuples([('Drivable area','mIoU'), ('Drivable area','FPS')#('Drivable area','Ti'),('Drivable area','Tf'),
    #                                                 ])
    # elif mode_lane:
    #     columns_Resume = pd.MultiIndex.from_tuples([('Lane marking','mIoU'), ('Lane marking','FPS')#,('Lane marking','Ti'),('Lane marking','Tf')
    #                                                 ])
    # else:
    #     print('Edit the code and configure other mode manualy! Exiting...')
    #     exit()

    # df3 = pd.DataFrame(columns=columns_Resume)

    # if mode_obj_dect:
    #     for tresh in ['50', '75']:
    #         ap_values = np.asarray(df2[('Object detection','AP@'+tresh)].tolist())
    #         ii = ~np.isnan(ap_values)
    #         m_ap = np.average(ap_values[ii])
    #         df3.loc['',('Object detection','mAP@'+tresh)] = m_ap


    # for t in ['Drivable area', 'Lane marking']:
    #     if (mode_drivable and t=='Drivable area') or (mode_lane and t=='Lane marking'):
    #         iou_values = np.asarray(df2[(t,'IoU')].tolist())
    #         ii = ~np.isnan(iou_values)
    #         m_iou = np.average(iou_values[ii])
    #         df3.loc['',(t,'mIoU')] = m_iou

    with pd.ExcelWriter(file_name + '.xlsx') as writer:
        # df3.to_excel(writer, sheet_name='Results')
        # df2.to_excel(writer, sheet_name='ByClass')
        df2.to_excel(writer, sheet_name='Results')
        df.to_excel(writer, sheet_name='Raw')
    
    print(f"Saved as {file_name}.xlsx")


# Meter na folha final o minimo das medias