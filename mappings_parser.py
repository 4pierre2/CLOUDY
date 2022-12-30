#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:56:35 2022

@author: pierre
"""

import numpy as np
import glob

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines

def get_info(filename):
    name = filename.split('/')[-1]#.split('_')[:3]
    a = name[0]
    n = name.split('_b')[0][3:].replace('_','.')
    b = name.split('_b')[1].split('_p')[0].split('_s')[0].replace('_', '.')
    n = float(n)
    if b == 'e':
        b = np.exp(1)
    b = float(b)
    lines = read_file(filename)
    hbs = 10**np.array(lines[-1].split(' - - -  ')[1].strip().split('   '), dtype='float')
    return filename, n, b, hbs

def clean_line(line):
    while '  ' in line:
        line = line.replace('  ', ' ')
    line = line.strip().split(' ')
    for k in range(len(line)):
        # print(line)
        if (k != 1) and (k != 2):
            line[k] = float(line[k])
            if k == 3:
                line[k] *= 1e-10
    return line


def clean_column_line(line):
    while '  ' in line:
        line = line.replace('  ', ' ')
    line = line.strip().split(' ')
    return line

def parse_file(filename, lam0 = 0.3e-6, lam1=10e-6):
    lines = read_file(filename)
    grids = []
    grids_infos = []
    parsed_infos = []
    vels = np.arange(100, 1001, 25)
    info_file = get_info(filename)
    lines_names = []
    lines_wavelengths = []
    array = []
    for l in lines:
        if ('#' not in l) and ('log(' not in l):
            cleaned_line = clean_line(l)
            lines_names.append(cleaned_line[1]+' '+cleaned_line[2])
            lines_wavelengths.append(str(1e6*cleaned_line[3])+'m')
            array.append(cleaned_line[4:])
    array = np.array(array).T
    
    for x in range(len(array)):
        parsed_lines = []
        for j in range(len(array[0])):
            wl = 1e-6*float(lines_wavelengths[j][:-1])
            if (wl > lam0) and (wl < lam1):
                parsed_lines.append([lines_names[j], lines_wavelengths[j], wl, info_file[-1][x]*array[x, j], array[x, j]])
        grids.append(parsed_lines)
        grids_infos.append([*info_file[:-1], vels[x]])
    return grids_infos, grids

def gauss(x, flux, x0, std):
    a = flux/(np.pi*2*std**2)**0.5
    return a*np.exp(-(x-x0)**2/(2*std**2))

def make_spec(lines, lambdas, std_sph, std_sinfo, lambda_break=1.78969e-6):
    mask_sph = lambdas < lambda_break
    mask_sinfo = lambdas > lambda_break
    spec = np.zeros(np.shape(lambdas))
    for l in lines:
        wl = l[2]
        fl = l[3]
        spec += gauss(lambdas, fl, wl, std_sph)*mask_sph
        spec += gauss(lambdas, fl, wl, std_sinfo)*mask_sinfo
    return spec
        
def parse_directory(directory, lam0 = 0.3e-6, lam1=10e-6):
    filenames = set(glob.glob(directory+'/*n*_sp*.txt')) - set(glob.glob(directory+'/R_*s*.txt')) - set(glob.glob(directory+'/J_*s*.txt')) - set(glob.glob(directory+'/P_*s*.txt')) - set(glob.glob(directory+'/Q_*s*.txt'))
    grids_infos = []
    grids = []
    for filename in filenames:
        # print(filename)
        grid_infos, grid = parse_file(filename, lam0=lam0, lam1=lam1)
        for j in range(len(grid_infos)):
            grids_infos.append(grid_infos[j])
            grids.append(grid[j])
    return grids_infos, grids
    
def parse_column_densities(filename, V):
    lines = read_file(filename)
    clean_lines = []
    go = False
    for m in range(len(lines)):
        l = lines[m]
        line_clean = clean_column_line(l)
        # print(line_clean[0], line_clean[0] == 'V'+str(int(V)))
        if line_clean[0] == 'V='+str(int(V)):
            go = True
        if line_clean[0] == 'V='+str(int(V+25)):
            go = False
        if go:
            clean_lines.append(line_clean)
    return clean_lines

def parse_results(prefix, V, path='./MAPPINGS/column_densities/'):
    filename_shock = path+'/'+prefix+'_shock_coldens.txt'
    filename_precursor = path+'/'+prefix+'_precursor_coldens.txt'
    return parse_column_densities(filename_precursor, V), parse_column_densities(filename_shock, V)
    
#%%

# filenames = glob.glob('./MAPPINGS/emission_line_ratios/*.txt')

# grids_infos = []
# grids = []
# for filename in filenames:
#     print(filename)
#     grid_infos, grid = parse_file(filename)
#     for j in range(len(grid_infos)):
#         grids_infos.append(grid_infos[j])
#         grids.append(grid[j])
    
# lambdas = np.arange(1e-6, 2.5e-6, 1e-9)
# sp = make_spec(grids[100], lambdas, 2.72e-9)