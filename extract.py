import re
import numpy as np
import copy
from functools import reduce
import itertools
import pickle


def get_deep(l, indices):
     if (len(indices) > 1) and isinstance(l[indices[0]], list):
         return get_deep(l[indices[0]], indices[1:])
     else:
         return l[indices[0]]

def set_deep(l, indices, value):
     if (len(indices) > 1):
         set_deep(l[indices[0]], indices[1:], value)
     else:
         l[indices[0]] = value
         
def remove_from_list(l):
    r = []
    for e in l:
        if isinstance(e, list):
            r.append(remove_from_list(e))
    return r

#%%

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines

def split_grids(lines):
    grids = []
    grid = []
    next_grid = False
    for l in lines:
        # if "GRID_DELIMIT" in l:
        if "        Cloudy 17.02" in l:
            grids.append(grid)
            grid=[]
        grid.append(l)
    grids.append(grid)
    grids = grids[3:]
    return grids
        

def get_raw_line_fluxes(grid):
    raw_line_fluxes = []
    emergent_line_has_appeared = False
    following_blank_line_has_appeared = False
    for l in grid:
        if emergent_line_has_appeared and (l == '\n'):
            following_blank_line_has_appeared = True
        if emergent_line_has_appeared and (not following_blank_line_has_appeared):
            raw_line_fluxes.append(l)
        if "        Emergent line intensities" in l:
            emergent_line_has_appeared = True
    return raw_line_fluxes

def get_cleaned_line_fluxes(raw_line_fluxes):
    clean_lines = []
    small_lines = []
    for k in range(3):
        for l in raw_line_fluxes:
            small_line = l[k*43:(k+1)*43]
            small_lines.append(small_line)
    for l in small_lines:
        if (not ('....' in l)) and  (not ('****' in l)):
            try:
                l = l.replace('\n', '')
                name = l[:10]
                wl = l[10:20].strip()
                if "A" in wl:
                    wl = float(wl.replace("A",""))*1e-10
                elif "m" in wl:
                    wl = float(wl.replace("m",""))*1e-6
                elif "c" in wl:
                    wl = float(wl.replace("m",""))*1e-2
                log = float(l[21:28])
                flux = float(l[29:])
                clean_lines.append([name, wl, log, flux])
            except:
                bvr = 3
    return clean_lines

def get_wl_flux_lines(clean_lines, wl_min=1e-10, wl_max=1e10, exclude=['TALL', 'Blnd']):
    names = []
    wls = []
    fls = []
    for l in clean_lines:
        wl = l[1]
        if (wl>wl_min) and (wl<wl_max):
            test = True
            for e in exclude:
                if e in l[0]:
                    test=False
            if test:
                fl = l[-1]
                name = l[0]
                names.append(name)
                wls.append(wl)
                fls.append(fl)
    return names, wls, fls

# def get_wls_of_interest(filename, wl_min=9.71e-7, wl_max=2.430e-6):
#     lines = read_file(filename)
#     grids = split_grids(lines)
#     wlss = []
#     for grid in grids:
#         raw_line_fluxes = get_raw_line_fluxes(grid)
#         clean_lines = get_cleaned_line_fluxes(raw_line_fluxes)
#         names, wls, fls = get_wl_flux_lines(clean_lines, wl_min=wl_min, wl_max=wl_max)
#         wlss.append(wls)
#     wlss_test = np.unique(np.around(1e12*np.array(wls)))
#     wlss = wlss_test*1e-12
#     return wlss

def get_infos_grid(grid, keywords):
    infos = {}
    for l in grid:
        for keyword in keywords:
            if keyword in l:
                split_line = re.split('('+keyword+')', l)
                i = split_line.index(keyword)
                value = float(split_line[i+1][:8])
                infos[keyword] = value
    return infos

def make_ranges(grids, keywords):
    ranges = []
    for keyword in keywords:
        values = []
        for grid in grids:
            values.append(get_infos_grid(grid, [keyword])[keyword])
        values = sorted(np.unique(values))
        ranges.append(values)
    return ranges

def make_array(filename, keywords):
    lines = read_file(filename)
    grids = split_grids(lines)
    
    ranges = make_ranges(grids, keywords)
    mesh = np.meshgrid(*ranges, indexing='ij')
    array = np.zeros(np.shape(mesh[0])).tolist()
    for grid in grids:
        infos = get_infos_grid(grid, keywords)
        indexs = []
        n = 0
        for keyword in keywords:
            index = np.argmin((np.array(ranges[n])-infos[keyword])**2)
            indexs.append(index)
            n+=1
        raw_line_fluxes = get_raw_line_fluxes(grid)
        clean_lines = get_cleaned_line_fluxes(raw_line_fluxes)
        names, wls, fls = get_wl_flux_lines(clean_lines, wl_min=9.71e-7, wl_max=2.43e-6)
        set_deep(array, indexs, [names, wls, fls])
        
    return array
    
        


filename = "./MODEL_2_extended_paral/model.out"
filename = "./MODEL_4/model.out"
lines = read_file(filename)
grids = split_grids(lines)

# raw_line_fluxes = get_raw_line_fluxes(grids[13])
# clean_lines = get_cleaned_line_fluxes(raw_line_fluxes)

keywords = ["BLACKbody= ", "HDEN="]
infos = get_infos_grid(grids[-1], keywords)
ranges = make_ranges(grids, keywords)

array = make_array(filename, keywords)

# pickle.dump(array, open("./MODEL_2_extended_paral/results/nested_list_cloudy.p", "wb"))
# pickle.dump(ranges, open("./MODEL_2_extended_paral/results/ranges_cloudy.p", "wb"))
