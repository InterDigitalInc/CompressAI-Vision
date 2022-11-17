# how do we pad or scale for different scales
vf_per_scale = {
    100: "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    75: "scale=ceil(iw*3/8)*2:ceil(ih*3/8)*2",
    50: "scale=ceil(iw/4)*2:ceil(ih/4)*2",
    25: "scale=ceil(iw/8)*2:ceil(ih/8)*2",
}
# how do we backscale for different scales:
inv_vf_per_scale = {
    100: "crop={width}:{height}:0:0",
    75: "scale={width}:{height}",
    50: "scale={width}:{height}",
    25: "scale={width}:{height}",
}
