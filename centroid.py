import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#Function to calculate the centroid
def find_centroid(array):
    t_weight = np.sum(array)
    
    row_sums = np.sum(array, axis=1)
    result = np.sum(row_sums * (np.arange(array.shape[0])+1))
    l = result / t_weight

    col_sums = np.sum(array, axis=0)
    result = np.sum(col_sums * (np.arange(array.shape[1])+1))
    p = result / t_weight

    return (l, p)

def display_centroid(array,flag):
    
    array_copy = array.copy()

    centroid = find_centroid(array_copy)

    if flag:
        st.subheader(f'Centroid:{centroid}')

        # Create a graph for the array
        fig = plt.figure(figsize=(15.0, 15.0))  

        #creating axis
        ax = fig.add_subplot()
        im = ax.imshow(array_copy, cmap='viridis', origin='lower')

        # Annotate each cell with the weight
        for i in range(array_copy.shape[0]):
            for j in range(array_copy.shape[1]):
                text = ax.text(float(j), float(i), f'{array_copy[i, j]:.5f}', ha='center', va='center', color='w')

        ax.set_xticks([])
        ax.set_yticks([])

        # Locating centroid
        ax.scatter(centroid[0]-1.5,centroid[1]-1.5, marker='o', color='red', label='Centroid', s=100)  # s adjusts the size of the marker

        ax.set_title('Array',fontsize=30)
        plt.colorbar(im, ax=ax)

        st.pyplot(plt)

    else:
        return centroid

# Streamlit UI
st.title('Centroid Variations in Response to Position and Intensity Changes')
st.sidebar.header('Controls')

if 'x' not in st.session_state:
    st.session_state.x = 0  #counter for each press

# Creating initial array (only once)
if 'array' not in st.session_state:
    st.session_state.array = np.random.uniform(250, 350, (10, 10))
    st.session_state.array[4][4] = 1500
    st.session_state.array[3][3] = 470
    st.session_state.array[3][4] = 750
    st.session_state.array[3][5] = 590
    st.session_state.array[4][3] = 700
    st.session_state.array[4][5] = 800
    st.session_state.array[5][3] = 500
    st.session_state.array[5][4] = 850
    st.session_state.array[5][5] = 450

# Button to add noise and display the graph
clicked = st.sidebar.button('Find centriod')

if clicked:
    display_centroid(st.session_state.array,1)

def add_noise_and_display(array):
    
    top_left = (3, 3)
    bottom_right = (5, 5)

    centroid=display_centroid(array,0)

    c=[]
    delp=[]
    dell=[]

    for k in range(10):

        st.subheader(f'Sample Number:{k+1}')
        array_copy = array.copy()
        noise = np.zeros((3, 3))

        for i in range(top_left[0], bottom_right[0] + 1):
            for j in range(top_left[1], bottom_right[1] + 1):
                random_value = np.random.uniform(-50, 50)
                noise[i-top_left[0]][j-top_left[1]] = random_value
                array_copy[i, j] += random_value

        c.append(find_centroid(array_copy))
        delp.append(abs(c[k][0]-centroid[0]))
        dell.append(abs(c[k][1]-centroid[1]))
        
        st.subheader(f'△Centroid:{(delp[k],dell[k])}')

        # Create a graph for the array
        fig = plt.figure(figsize=(15.0, 15.0))  

        #creating axis
        ax = fig.add_subplot()
        im = ax.imshow(array_copy, cmap='viridis', origin='lower')

        # Annotate each cell with the weight
        for i in range(array_copy.shape[0]):
            for j in range(array_copy.shape[1]):
                text = ax.text(float(j), float(i), f'{array_copy[i, j]:.5f}', ha='center', va='center', color='w')

        ax.set_xticks([])
        ax.set_yticks([])

        # Locating centroid
        ax.scatter(c[k][0]-1.5,c[k][1]-1.5, marker='o', color='red', label='Centroid', s=100)  # s adjusts the size of the marker

        ax.set_title('Array',fontsize=30)
        plt.colorbar(im, ax=ax)

        st.pyplot(plt)

        #Graph for noise
        fig1 = plt.figure(figsize=(5, 5))  

        ax1= fig1.add_subplot(111)
        im1 = ax1.imshow(noise, cmap='viridis', origin='lower')
    
        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                text = ax1.text(j, i, f'{noise[i, j]:.5f}', ha='center', va='center', color='w')

        ax1.set_xticks([])
        ax1.set_yticks([])

        ax1.set_title('Noise')
        plt.colorbar(im1, ax=ax1)
        st.pyplot(plt)
        st.write()

        noise_file='noise_'+str(k)+'.txt'
        centroid_file='centriod_'+str(k)+'.txt'
        #np.savetxt(noise_file,noise,fmt="%.5f" ,delimiter=',')
        #np.savetxt(centroid_file,c[k],fmt="%.5f" ,delimiter=',')

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), dell, marker='o', linestyle='-', color='blue', label='delL')
    plt.plot(range(1, 11), delp, marker='o', linestyle='-', color='green', label='delP')
    plt.xlabel('Sample Number')
    plt.ylabel('delL / delP')
    plt.title('delL and delP vs Sample Number')
    plt.legend()
    st.pyplot(plt)
    
    
clicked = st.sidebar.button('Add Noise and find centriod')

if clicked:
    add_noise_and_display(st.session_state.array)

def modify_corner(array):

    location=(1,1)
    centroid=display_centroid(array,0)
    st.title(f'Modified location:{(location[0]+1,location[1]+1)}')

    c=[]
    delp=[]
    dell=[]
    intensity=[]

    for k in range(10):

        st.subheader(f'Sample Number:{k+1}')
        array_copy = array.copy()
        i, j = location
        array_copy[i, j] = np.random.randint(0, 4096)  # Random value between 0 and 4095
        intensity.append(array_copy[1,1])

        c.append(find_centroid(array_copy))
        delp.append(abs(c[k][0]-centroid[0]))
        dell.append(abs(c[k][1]-centroid[1]))
        
        st.subheader(f'△Centroid:{(delp[k],dell[k])}')

    # Creating graph for modification at each location
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(111)
        im = ax.imshow(array_copy, cmap='viridis', origin='lower')

    # Annotate each cell with the weight
        for i in range(array_copy.shape[0]):
            for j in range(array_copy.shape[1]):
                text = ax.text(j, i, f'{array_copy[i, j]:.5f}', ha='center', va='center', color='w')

        ax.set_xticks([])
        ax.set_yticks([])

    # Locate the centroid
        centroid_x, centroid_y = c[k][1]-1.5, c[k][0]-1.5
        ax.scatter(centroid_x, centroid_y, marker='o', color='red', label='Centroid', s=100)  

        ax.set_title(f'Modified Array with Weights')
        plt.colorbar(im, ax=ax)

        st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), dell, marker='o', linestyle='-', color='blue', label='delL')
    plt.plot(range(1, 11), delp, marker='o', linestyle='-', color='green', label='delP')
    plt.xlabel('Sample Number')
    plt.ylabel('delL / delP')
    plt.title('delL and delP vs Sample Number')
    plt.legend()

    # Annotate the points with intensity values
    for i, txt in enumerate(intensity):
        plt.annotate(f'Intensity: {txt}', (i + 1, dell[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='red')

    st.pyplot(plt)

#Button to modify corners and display graph
clicked = st.sidebar.button('Modify corners and find centriod')

if clicked:
    modify_corner(st.session_state.array)
 
def modify_corner_high(array):

    location=(1,1)
    centroid=display_centroid(array,0)
    st.title(f'Modified location:{(location[0]+1,location[1]+1)}')

    c=[]
    delp=[]
    dell=[]
    intensity=[]

    for k in range(10):

        st.subheader(f'Sample Number:{k+1}')
        array_copy = array.copy()
        i, j = location
        array_copy[i, j] = np.random.randint(3000, 4096)  # Random value between 0 and 4095
        intensity.append(array_copy[1,1])

        c.append(find_centroid(array_copy))
        delp.append(abs(c[k][0]-centroid[0]))
        dell.append(abs(c[k][1]-centroid[1]))
        
        st.subheader(f'△Centroid:{(delp[k],dell[k])}')

    # Creating graph for modification at each location
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(111)
        im = ax.imshow(array_copy, cmap='viridis', origin='lower')

    # Annotate each cell with the weight
        for i in range(array_copy.shape[0]):
            for j in range(array_copy.shape[1]):
                text = ax.text(j, i, f'{array_copy[i, j]:.5f}', ha='center', va='center', color='w')

        ax.set_xticks([])
        ax.set_yticks([])

    # Locate the centroid
        centroid_x, centroid_y = c[k][1]-1.5, c[k][0]-1.5
        ax.scatter(centroid_x, centroid_y, marker='o', color='red', label='Centroid', s=100)  

        ax.set_title(f'Modified Array with Weights')
        plt.colorbar(im, ax=ax)

        st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), dell, marker='o', linestyle='-', color='blue', label='delL')
    plt.plot(range(1, 11), delp, marker='o', linestyle='-', color='green', label='delP')
    plt.xlabel('Sample Number')
    plt.ylabel('delL / delP')
    plt.title('delL and delP vs Sample Number')
    plt.legend()

    # Annotate the points with intensity values
    for i, txt in enumerate(intensity):
        plt.annotate(f'Intensity: {txt}', (i + 1, dell[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='red')

    st.pyplot(plt)


#Button to modify corners and display graph
clicked = st.sidebar.button('Modify corners(3000-4095) and find centriod')

if clicked:
    modify_corner_high(st.session_state.array)
 
def vary_position(array,a,b):

    locations=[(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9)]
    centroid=display_centroid(array,0)

    c=[]
    delp=[]
    dell=[]
    k=0

    value=np.random.randint(a,b)

    for location in locations:

        st.title(f'Modified location:{(location[0],location[1])}')
        array_copy = array.copy()
        i, j = location[0]-1,location[1]-1
        array_copy[i, j] = value  # Random value between 0 and 4095

        c.append(find_centroid(array_copy))
        delp.append(abs(c[k][0]-centroid[0]))
        dell.append(abs(c[k][1]-centroid[1]))
        
        st.subheader(f'△Centroid:{(delp[k],dell[k])}')

    # Creating graph for modification at each location
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(111)
        im = ax.imshow(array_copy, cmap='viridis', origin='lower')

    # Annotate each cell with the weight
        for i in range(array_copy.shape[0]):
            for j in range(array_copy.shape[1]):
                text = ax.text(j, i, f'{array_copy[i, j]:.5f}', ha='center', va='center', color='w')

        ax.set_xticks([])
        ax.set_yticks([])

    # Locate the centroid
        centroid_x, centroid_y = c[k][1]-1.5, c[k][0]-1.5
        ax.scatter(centroid_x, centroid_y, marker='o', color='red', label='Centroid', s=100)  

        ax.set_title(f'Modified Array with Weights')
        plt.colorbar(im, ax=ax)

        st.pyplot(plt)
        k=k+1

    sorted_dell = [x for _, x in sorted(zip(range(len(dell)), dell), reverse=True)]
    sorted_locations = [x for _, x in sorted(zip(range(len(locations)), locations), reverse=True)]
    sorted_delp = [x for _, x in sorted(zip(range(len(delp)), delp), reverse=True)]

    plt.figure(figsize=(8, 6))
    plt.plot([location[0] for location in sorted_locations], sorted_dell, marker='o', linestyle='-', color='red',label='delL')
    plt.plot([location[0] for location in sorted_locations], sorted_delp, marker='o', linestyle='-', color='blue',label='delP')
    plt.xlabel('Position')
    plt.ylabel('delL/delP')
    plt.title(f'delL and delP vs Position, Intensity:{value}')
    plt.legend()
    st.pyplot(plt)
    

st.sidebar.subheader('Varying position keeping intensity fixed')

clicked = st.sidebar.button('S1:0-1000')
if clicked:
    vary_position(st.session_state.array,0,1000)

clicked = st.sidebar.button('S2:1000-2000')
if clicked:
    vary_position(st.session_state.array,1000,2000)

clicked = st.sidebar.button('S3:2000-3000')
if clicked:
    vary_position(st.session_state.array,2000,3000)

clicked = st.sidebar.button('S4:3000-4095')
if clicked:
    vary_position(st.session_state.array,3000,4096)


def gradient_generation(array,a,b):

    locations=[(1,8),(2,8),(1,7)]
    centroid=display_centroid(array,0)

    c=[]
    delp=[]
    dell=[]

    for k in range(10):

        array_copy = array.copy()

        for location in locations:
            i, j = location
            array_copy[i, j] = np.random.randint(a,b)  # Random value between 0 and 4095

        c.append(find_centroid(array_copy))
        delp.append(abs(c[k][0]-centroid[0]))
        dell.append(abs(c[k][1]-centroid[1]))
        
        st.subheader(f'△Centroid:{(delp[k],dell[k])}')

    # Creating graph for modification at each location
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(111)
        im = ax.imshow(array_copy, cmap='viridis', origin='lower')

    # Annotate each cell with the weight
        for i in range(array_copy.shape[0]):
            for j in range(array_copy.shape[1]):
                text = ax.text(j, i, f'{array_copy[i, j]:.5f}', ha='center', va='center', color='w')

        ax.set_xticks([])
        ax.set_yticks([])

    # Locate the centroid
        centroid_x, centroid_y = c[k][1]-1.5, c[k][0]-1.5
        ax.scatter(centroid_x, centroid_y, marker='o', color='red', label='Centroid', s=100)  

        ax.set_title(f'Modified Array with Weights')
        plt.colorbar(im, ax=ax)

        st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1,11), dell, marker='o', linestyle='-', color='blue',label="delL")
    plt.plot(range(1,11), delp, marker='o', linestyle='-', color='red',label='delP')
    plt.xlabel('sample_number')
    plt.ylabel('delL/delP')
    plt.title('delL and delP vs sample_number')
    plt.legend()
    st.pyplot(plt)
    
st.sidebar.subheader('Gradient Generation')

clicked = st.sidebar.button('s1:0-1000')
if clicked:
    gradient_generation(st.session_state.array,0,1000)

clicked = st.sidebar.button('s2:1000-2000')
if clicked:
    gradient_generation(st.session_state.array,1000,2000)

clicked = st.sidebar.button('s3:2000-3000')
if clicked:
    gradient_generation(st.session_state.array,2000,3000)

clicked = st.sidebar.button('s4:3000-4095')
if clicked:
    gradient_generation(st.session_state.array,3000,4096)


def gradient_varying_blocks(array,a,b):

    locations=[[(1,8)],[(1,7),(2,8)],[(1,6),(2,7),(3,8)],[(1,5),(2,6),(3,7),(4,8)],[(1,4),(2,5),(3,6),(4,7),(5,8)]]
    centroid=display_centroid(array,0)

    c=[]
    delp=[]
    dell=[]
    k=0

    array_copy = array.copy()

    for location in locations:

        for l in location:
            i, j = l
            array_copy[i, j] = np.random.randint(a,b)  # Random value between 0 and 4095

        c.append(find_centroid(array_copy))
        delp.append(abs(c[k][0]-centroid[0]))
        dell.append(abs(c[k][1]-centroid[1]))
        
        st.subheader(f'△Centroid:{(delp[k],dell[k])}')

    # Creating graph for modification at each location
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(111)
        im = ax.imshow(array_copy, cmap='viridis', origin='lower')

    # Annotate each cell with the weight
        for i in range(array_copy.shape[0]):
            for j in range(array_copy.shape[1]):
                text = ax.text(j, i, f'{array_copy[i, j]:.5f}', ha='center', va='center', color='w')

        ax.set_xticks([])
        ax.set_yticks([])

    # Locate the centroid
        centroid_x, centroid_y = c[k][1]-1.5, c[k][0]-1.5
        ax.scatter(centroid_x, centroid_y, marker='o', color='red', label='Centroid', s=100)  

        ax.set_title(f'Modified Array with Weights')
        plt.colorbar(im, ax=ax)

        st.pyplot(plt)
        k=k+1

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(locations)), dell, marker='o', linestyle='-', color='blue',label='delL')
    plt.plot(range(len(locations)), delp, marker='o', linestyle='-', color='red',label='delP')
    plt.xlabel('sample_number')
    plt.ylabel('delL/delP')
    plt.title('delL/delP vs sample_number')
    plt.legend()
    st.pyplot(plt)
    
st.sidebar.subheader('Gradient Energy in varying blocks')

clicked = st.sidebar.button('1:0-1000')
if clicked:
    gradient_varying_blocks(st.session_state.array,0,1000)

clicked = st.sidebar.button('2:1000-2000')
if clicked:
    gradient_varying_blocks(st.session_state.array,1000,2000)

clicked = st.sidebar.button('3:2000-3000')
if clicked:
    gradient_varying_blocks(st.session_state.array,2000,3000)

clicked = st.sidebar.button('4:3000-4095')
if clicked:
    gradient_varying_blocks(st.session_state.array,3000,4096)
