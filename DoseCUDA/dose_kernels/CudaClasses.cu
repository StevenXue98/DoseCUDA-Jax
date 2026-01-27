#include "./CudaClasses.cuh"


__host__ CudaBeam::CudaBeam(CudaBeam * h_beam){

	this->iso = h_beam->iso;
	this->src = h_beam->src;
	this->gantry_angle = h_beam->gantry_angle;
	this->couch_angle = h_beam->couch_angle;
	this->singa = h_beam->singa;
	this->cosga = h_beam->cosga;
	this->sinta = h_beam->sinta;
	this->costa = h_beam->costa;

}

__host__ CudaBeam::CudaBeam(float * iso, float gantry_angle, float couch_angle, float src_dist){

	this->iso.x = iso[0];
	this->iso.y = iso[1];
	this->iso.z = iso[2];
	this->gantry_angle = gantry_angle;
	this->couch_angle = couch_angle;

	//starting coordinate is [x, y, z] = [0.0, SAD, 0.0]
	//machine angles in radians
	float ga = gantry_angle * CUDART_PI_F / 180.0f;
	float ta = couch_angle * CUDART_PI_F / 180.0f;
	this->singa = sinf(ga);
	this->cosga = cosf(ga);
	this->sinta = sinf(ta);
	this->costa = cosf(ta);

	//gantry rotation - rotate about z-axis
	float xg, yg;
	xg = -src_dist * this->singa;
	yg = src_dist * this->cosga;
	//zg = 0.0;

	//table rotation - rotate about y-axis
	float xt, yt, zt;
	xt = xg * this->costa;
	yt = yg;
	zt = -xg * this->sinta;

	//translate based on iso location in image
	this->src.x = xt;
	this->src.y = yt;
	this->src.z = zt;

}

__device__ float CudaBeam::distanceToSource(const PointXYZ * point_xyz){
	
	float dx = this->src.x - point_xyz->x;
	float dy = this->src.y - point_xyz->y;
	float dz = this->src.z - point_xyz->z;

	float r = norm3df(dx, dy, dz);

	return r;

}

__device__ void CudaBeam::pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head){

	float sinx, cosx;

	//table rotation - rotate about y-axis
	float xt, yt, zt;
	sinx = -this->sinta;
	cosx = this->costa;
	xt = point_img->x * cosx + point_img->z * sinx;
	yt = point_img->y;
	zt = -point_img->x * sinx + point_img->z * cosx;

	//gantry rotation - rotate about z-axis
	float xg, yg, zg;
	sinx = -this->singa;
	cosx = this->cosga;
	xg = xt * cosx - yt * sinx;
	yg = xt * sinx + yt * cosx;
	zg = zt;


	//swap final coordinates to match DICOM nozzle coordinate system
	//for an AP beam:
	//	beam travels in negative z direction
	//	positive x is to the patient's left
	//	positive y is to the patient's superior
	point_head->x = -xg;
	point_head->y = zg;
	point_head->z = yg;

}

__device__ void CudaBeam::pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img){

	float sinx, cosx;

	//convert back to DICOM patient LPS coordinates
	float xz = -point_head->x;
	float yz = point_head->z;
	float zz = point_head->y;

	//gantry rotation - rotate about z-axis (negative direction)
	float xg, yg, zg;
	sinx = this->singa;
	cosx = this->cosga;
	xg = xz * cosx - yz * sinx;
	yg = xz * sinx + yz * cosx;
	zg = zz;

	//table rotation - rotate about y-axis (negative direction)
	float xt, yt, zt;
	sinx = this->sinta;
	cosx = this->costa;
	xt = xg * cosx + zg * sinx;
	yt = yg;
	zt = -xg * sinx + zg * cosx;

	point_img->x = xt;
	point_img->y = yt;
	point_img->z = zt;

}

__device__ void CudaBeam::pointXYZClosestCAXPoint(const PointXYZ * point_xyz, PointXYZ * point_cax){

	float d1x = this->iso.x - this->src.x;
	float d1y = this->iso.y - this->src.y;
	float d1z = this->iso.z - this->src.z;

	float d2x = point_xyz->x - this->src.x;
	float d2y = point_xyz->y - this->src.y;
	float d2z = point_xyz->z - this->src.z;

	float t = (d1x * d2x + d1y * d2y + d1z * d2z) / (d1x * d1x + d1y * d1y + d1z * d1z);

	point_cax->x = fmaf(t, d1x, this->src.x);
	point_cax->y = fmaf(t, d1y, this->src.y);
	point_cax->z = fmaf(t, d1z, this->src.z);

}

__device__ float CudaBeam::pointXYZDistanceToCAX(const PointXYZ * point_head_xyz){

	return hypotf(point_head_xyz->x, point_head_xyz->z);

}

__device__ float CudaBeam::pointXYZDistanceToSource(const PointXYZ * point_img_xyz){

	float dx = this->src.x - point_img_xyz->x;
	float dy = this->src.y - point_img_xyz->y;
	float dz = this->src.z - point_img_xyz->z;

	return norm3df(dx, dy, dz);

}

__host__ CudaDose::CudaDose(CudaDose * h_dose){
	this->img_sz = h_dose->img_sz;
	this->spacing = h_dose->spacing;
	this->num_voxels = h_dose->num_voxels;
}

__host__ CudaDose::CudaDose(size_t img_sz[], float spacing) {

	this->img_sz.i = img_sz[2];
	this->img_sz.j = img_sz[1];
	this->img_sz.k = img_sz[0];

	this->spacing = spacing;

	this->num_voxels = this->img_sz.i * this->img_sz.j * this->img_sz.k;

}

__global__ void rayTraceKernel(CudaDose * dose, CudaBeam * beam, Texture3D DensityTexture){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	int vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);

	PointXYZ uvec;
	beam->unitVectorToSource(&vox_xyz, &uvec);

	PointXYZ vox_ray_xyz, tex_xyz;

    float ray_length = 0.0f;
    float wet_sum = -0.05f;
    float density = 0.0f;
	const float step_length = 1.0f;

    while(true){

		vox_ray_xyz.x = fmaf(uvec.x, ray_length, vox_xyz.x);
		vox_ray_xyz.y = fmaf(uvec.y, ray_length, vox_xyz.y);
		vox_ray_xyz.z = fmaf(uvec.z, ray_length, vox_xyz.z);

		dose->pointXYZtoTextureXYZ(&vox_ray_xyz, &tex_xyz, beam);
		if (!dose->textureXYZWithinImage(&tex_xyz)) {
			break;
		}
		density = DensityTexture.sample(tex_xyz);

		wet_sum = fmaf(fmaxf(density, 0.0f), step_length / 10.0f, wet_sum);

		ray_length += step_length;

	}

	dose->WETArray[vox_index] = wet_sum;

    __syncthreads();

}
