import Ma223_BE_lib


Ma223_BE_lib.plot_derivate("sphere1.jpg", "sphere2.jpg")
#%%

Ma223_BE_lib.create_video_vector('Basketball', '', Ma223_BE_lib.MC_flux_calc ,'vector_video.mp4')
Ma223_BE_lib.create_video_vector('Basketball', '', Ma223_BE_lib.MC_flux_calc_v1 ,'vector_video.mp4')
Ma223_BE_lib.create_video_vector('Basketball', '', Ma223_BE_lib.MC_flux_calc_v2 ,'vector_video.mp4')
#%%

Ma223_BE_lib.create_video_vector('Basketball', '', Ma223_BE_lib.var_flux_calc ,'vector_video.mp4')
Ma223_BE_lib.create_video_vector('Basketball', '', Ma223_BE_lib.var_flux_calc_v2 ,'vector_video.mp4')
#%%

Ma223_BE_lib.create_video_mask('Basketball', '' ,'masked_video.mp4')
Ma223_BE_lib.track_point('Basketball', '', 'tracked_video.mp4', Ma223_BE_lib.var_flux_calc_v2)
#%%
