% Create 3D saliency ground truth by mapping the corresponding 2D saliency map onto the surface of the 3D mesh 
% while considering the visibility of each 3D vertex with regard to the corresponding viewpoint.
% Copyright (c) 2023 Ran Song

shapefolder = '.\mesh3d'; % The path of the folder where the 3D meshes are.
viewfolder = '.\view2d'; % The path of the folder where the 2D views of the 3D meshes are.
saliencyfolder = '.\saliency3d_map'; % The path of the folder where the visualised 3D saliency maps are saved.
gtfolder = '.\saliency3d_gt'; % The path of the folder where the 3D saliency ground truth are saved.
viewpointfolder = '.\viewpoints'; % The path of the folder where the viewpoint files are.
viewsafolder = '.\saliency2d'; % The path of the folder where the 2D saliency maps of the 2D views are.

nViews = 3; % The 3D-ViSa dataset provides the saliency maps at 3 views for each 3D mesh.
outputSize = 1200; % The 2D views are in the size of 1200x1200.

list_viewsa = dir(viewsafolder);
list_viewsa(1:2)=[]; 
num_viewsa = numel(list_viewsa);

% Read in the 2D saliency maps.
gtsa = zeros(num_viewsa,outputSize,outputSize,3,'single');
for i = 1:num_viewsa
    I = imread([viewsafolder '\' list_viewsa(i).name]);
    gtsa(i,:,:,1) = I;
    gtsa(i,:,:,2) = I;
    gtsa(i,:,:,3) = I;
end

shapedir = dir(shapefolder);
viewdir = dir(viewfolder);
num_shape = numel(shapedir);

for i = 3:num_shape
    
    % Load and normalise the mesh.
    mesh = loadMesh( [shapedir(i).folder '\' shapedir(i).name] );
    viewpoints = load([viewpointfolder '\' shapedir(i).name(1:end-4) '.txt']);

    xn1=max(mesh.V(1,:));
    xn2=min(mesh.V(1,:));
    yn1=max(mesh.V(2,:));
    yn2=min(mesh.V(2,:));
    zn1=max(mesh.V(3,:));
    zn2=min(mesh.V(3,:));
    bbox=sqrt((xn1-xn2).^2+(yn1-yn2).^2+(zn1-zn2).^2);
    
    mesh.V(1,:)=double(mesh.V(1,:)-0.5*(xn1+xn2));
    mesh.V(2,:)=double(mesh.V(2,:)-0.5*(yn1+yn2));
    mesh.V(3,:)=double(mesh.V(3,:)-0.5*(zn1+zn2));
    mesh.F=double(mesh.F);
 
    p=mesh.V';
    t=mesh.F';
    
    imsvsa=zeros(length(p),nViews);
    
    imageidxs = (i-3)*nViews+1:(i-2)*nViews;
    ims = zeros(nViews, outputSize, outputSize,3,'uint8');
    
    for j = 1:nViews
        image_name = [viewfolder '\' viewdir(imageidxs(j)+2).name];
        ims(j,:,:,:) = imread(image_name);
    end
    
    imsa = gtsa(imageidxs,:,:,:);

    visibility = zeros(length(p),nViews);

    for ii=1:nViews
        az=viewpoints(ii,1);
        el=viewpoints(ii,2);

        % Crop the images and prepare for the 2D-to-3D saliency transfer.
        [crop,r1,r2,c1,c2]=autocrop(reshape(ims(ii,:,:,:),[outputSize outputSize 3]));
        
        aa=size(crop,1);
        bb=size(crop,2);
        longside=max(aa,bb);
        shortside=min(aa,bb);
        
        image1=imsa(ii,:,:,:);
        image1_sa=reshape(image1,[outputSize,outputSize,3]);
        image1_sa_norm = image1_sa;
        image1_sa_norm(image1_sa<0)=0;
        image1_sa_norm = (image1_sa_norm-min(image1_sa_norm(:)))./(max(image1_sa_norm(:))-min(image1_sa_norm(:)));
        ddd = rgb2gray(image1_sa_norm);
        
        % Conduct the spatial mapping.
        T=viewmtx(az,el);
        
        x4d=[p';ones(1,length(p))];
        x2d = T*x4d;

        xxa=max(x2d(1,:));
        xxi=min(x2d(1,:));
        yya=max(x2d(2,:));
        yyi=min(x2d(2,:));
        
        xx=xxa-xxi;
        yy=yya-yyi;
        
        longx=max(xx,yy);
        shortx=min(xx,yy);
        
        scale1=longside/longx;
        scale2=shortside/shortx;
        
        sscale=0.5*(scale1+scale2);
        x1=x2d(1,:)*sscale;
        y1=x2d(2,:)*sscale;
        
        x2=x1+0.5*outputSize;
        y2=y1-0.5*outputSize;
        
        cropsa = ddd(r1:r2,c1:c2);
        cropsa = imgaussfilt(cropsa,10);
        
        row_c = size(cropsa,1)/2;
        col_c = size(cropsa,2)/2;
        
        y2=-y2;
        x2=x2-min(x2);
        y2=y2-min(y2); % y2 is row;
        
        [vpy,vpx,vpz]=sph2cart(pi*viewpoints(ii,1)/180,pi*viewpoints(ii,2)/180,bbox);
        vpy=-vpy;
        
        % Use the smoothing function to slightly dilate the visible area.
        visibility_v = mark_visible_vertices(p,t,[vpx,vpy,vpz]);
        visibility_v = perform_mesh_smoothing(t,p,visibility_v);
        visibility(:,ii) = visibility_v;
        
        [impointsx,impointsy]=meshgrid(1:bb,1:aa);
        impoints=[impointsx(:) impointsy(:)];
        
        visible=find(visibility_v~=0);
        vx2=x2(visible);
        vy2=y2(visible);
        x2ddd=[vx2(:) vy2(:) ];
        ind_cor = knnsearch(impoints,x2ddd);
        
        % Assign the saliency of a pixel to a visible 3D vertex.
        for jj=1:length(visible)
            row=impoints(ind_cor(jj),2);
            col=impoints(ind_cor(jj),1);
            imsvsa(visible(jj),ii)=exp(cropsa(row,col));
        end
        
        saa = imsvsa(:,ii);
        saa(isnan(saa))=min(saa);
        saa = saa - min(saa);
        saa=(saa-min(saa))/(max(saa)-min(saa));
        imsvsa(:,ii) = saa;
        
        figure,trisurf(mesh.F',mesh.V(1,:),mesh.V(2,:),mesh.V(3,:),imsvsa(:,ii).^3);material dull;axis equal tight;axis off;shading interp;view(az,el);lightangle(az,el);lighting flat;colormap jet;
        saveas(gcf,fullfile(saliencyfolder,strcat(shapedir(i).name(1:end-4),num2str(ii),'.png')));
        writematrix(imsvsa(:,ii),fullfile(gtfolder,strcat(shapedir(i).name(1:end-4),num2str(ii),'.txt')));
        
        close all;

    end
end