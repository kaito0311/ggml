#define _USE_MATH_DEFINES        // for M_PI
#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// default hparams (ViT-B 32)

struct openai_vit_b32_hparams
{
    int32_t n_layer = 12;
    int32_t width = 768;
    int32_t head_width = 64;
    int32_t n_head = 12;
    int32_t patch_size = 32;
    int32_t img_size = 224;
    int32_t output_dim = 512;

    int32_t n_attention_pool_queries = 8;
    int32_t n_attention_pool = 256;

    int32_t ftype = 1; // float 32

    float mlp_ratio = 4.0;

    int32_t n_img_size() const { return img_size; }
    int32_t n_patch_size() const { return patch_size; }
    int32_t n_head_dim() const { return head_width; }
    int32_t n_img_bed() const { return n_img_size() / n_patch_size(); }
};

struct multihead_attention
{
    // struct ggml_tensor * ln_1_g;
    // struct ggml_tensor * ln_1_b;

    // struct ggml_tensor * ln_2_g;
    // struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor *c_attn_attn_w;
    struct ggml_tensor *c_attn_attn_b;

    struct ggml_tensor *c_attn_proj_w;
    struct ggml_tensor *c_attn_proj_b;

    // // mlp
    // struct ggml_tensor *c_mlp_fc_w;
    // struct ggml_tensor *c_mlp_fc_b;

    // struct ggml_tensor *c_mlp_proj_w;
    // struct ggml_tensor *c_mlp_proj_b;
};

// struct layer_scale {
//     struct ggml_tensor * gamma_w;
//     struct ggml_tensor * gamma_b;
// };

struct residual_attetion_block
{
    struct ggml_tensor *ln_1_w;
    struct ggml_tensor *ln_1_b;

    struct multihead_attention attn;

    // struct layer_scale * ls_1;  // not use

    struct ggml_tensor *ln_2_w;
    struct ggml_tensor *ln_2_b;

    // mlp
    struct ggml_tensor *fc_w;
    struct ggml_tensor *fc_b;

    struct ggml_tensor *proj_w;
    struct ggml_tensor *proj_b;

    // struct layer_scale * ls_2;  // not use
};

struct transformer_layer
{
    std::vector<residual_attetion_block> resblocks;
};

struct attention_pooler
{

    struct ggml_tensor *query; // nn.Parameter(torch.randn(n_queries, d_model))

    struct multihead_attention *attn;

    struct ggml_tenosr *ln_q_g;
    struct ggml_tenosr *ln_q_b;

    struct ggml_tenosr *ln_k_g;
    struct ggml_tenosr *ln_k_b;
};

struct vision_transformer
{

    struct ggml_tensor *conv1_w;
    // struct ggml_tensor *conv1_b; // bias = False

    struct ggml_tensor *class_embedding;
    struct ggml_tensor *position_embedding;

    struct transformer_layer transformer_block;

    // struct attention_pooler attn_pool;

    struct ggml_tensor *ln_post_w;
    struct ggml_tensor *ln_post_b;

    struct ggml_tensor *ln_pre_w;
    struct ggml_tensor *ln_pre_b;

    struct ggml_tensor *proj;
};

struct openclip_visual_model
{
    openai_vit_b32_hparams hparams;

    vision_transformer visual_enc;

    //
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// RGB uint8 imag. Reference "sam" example
struct image_u8
{
    int nx;
    int ny;

    std::vector<uint8_t> data;
};

// RGB float32 image
// Memory layout: RGBRGBRGB...
// Reference "sam" example
struct image_f32
{
    int nx;
    int ny;

    std::vector<float> data;
};

struct visual_clip_params
{
    int32_t seed = -1; // RNG see d
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    std::string model = "models/ggml_open_clip.bin";
    std::string fname_inp = "img.jpg";
    std::string fname_out = "img.put";
};

void print_t_f32(const char *title, struct ggml_tensor *t, int n = 10)
{
    printf("%s\n", title);
    float *data = (float *)t->data;
    printf("dims: % " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("First & Last %d elements:\n", n);
    for (int i = 0; i < std::min((int)(t->ne[0] * t->ne[1]), n); i++)
    {
        printf("%.5f ", data[i]);
        if (i != 0 && i % t->ne[0] == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
    for (int i = 0; i < std::min((int)(t->ne[0] * t->ne[1]), n); i++)
    {
        printf("%.5f ", data[ggml_nelements(t) - n + i]);
        if ((ggml_nelements(t) - n + i) % t->ne[0] == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
    double sum = 0.0;
    for (int i = 0; i < ggml_nelements(t); i++)
    {
        sum += data[i];
    }
    printf("sum:  %f\n\n", sum);
}

// load the model's weights from a file
bool open_clip_model_load(
    const visual_clip_params &params,
    openclip_visual_model &model)
{
    fprintf(stderr, "%s: loading model from '%s' - please wait ... \n", __func__, params.model.c_str());

    auto fin = std::ifstream(params.model, std::ios::binary);

    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s' \n", __func__, params.model.c_str());
        return false;
    }

    // verify magic term
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, params.model.c_str());
            return false;
        }
    }

    // load hparams:
    {
        // override hyperparamer
        auto &hparams = model.hparams;

        fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *)&hparams.width, sizeof(hparams.width));
        fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_attention_pool_queries, sizeof(hparams.n_attention_pool_queries));
        fin.read((char *)&hparams.n_attention_pool, sizeof(hparams.n_attention_pool));
        fin.read((char *)&hparams.patch_size, sizeof(hparams.patch_size));
        fin.read((char *)&hparams.img_size, sizeof(hparams.img_size));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_layer                     = %d \n", __func__, hparams.n_layer);
        printf("%s: width                       = %d \n", __func__, hparams.width);
        printf("%s: n_head                      = %d \n", __func__, hparams.n_head);
        printf("%s: n_attention_pool_queries    = %d \n", __func__, hparams.n_attention_pool_queries);
        printf("%s: n_attention_pool            = %d \n", __func__, hparams.n_attention_pool);
        printf("%s: patch_size                  = %d \n", __func__, hparams.patch_size);
        printf("%s: img_size                    = %d \n", __func__, hparams.img_size);
        printf("%s: ftype                       = %d \n", __func__, hparams.ftype);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT)
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, params.model.c_str(), model.hparams.ftype);
        return false;
    }

    auto &ctx = model.ctx;

    const size_t ctx_size = [&]()
    {
        size_t ctx_size = 0;

        const auto &hparams = model.hparams;

        const int32_t n_layer = hparams.n_layer;
        const int32_t dim_embed = hparams.width;
        const int32_t n_head = hparams.n_head;
        const int32_t patch_size = hparams.patch_size;
        const int32_t head_width = hparams.head_width;
        const int32_t img_size = hparams.img_size;
        const int32_t output_dim = hparams.output_dim;
        const int32_t mlp_width = hparams.mlp_ratio * dim_embed;

        const int32_t n_img_emb = hparams.n_img_bed();
        const int32_t grid_size = img_size / patch_size;

        {
            // class embed
            ctx_size += dim_embed * 1 * ggml_type_size(GGML_TYPE_F32);

            // position embed
            ctx_size += dim_embed * (grid_size * grid_size + 1) * ggml_type_size(GGML_TYPE_F32);

            // proj
            ctx_size += dim_embed * output_dim * ggml_type_size(GGML_TYPE_F32);

            // patch embed
            ctx_size += patch_size * patch_size * 3 * dim_embed * ggml_type_size(GGML_TYPE_F32);

            // ln pre
            ctx_size += 2 * dim_embed * ggml_type_size(GGML_TYPE_F32); // both weight + bias

            // transformer layers
            // in_proj weight + bias
            ctx_size += n_layer * 3 * dim_embed * dim_embed * ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_layer * 3 * dim_embed * 1 * ggml_type_size(GGML_TYPE_F32);

            // out_proj weight + bias
            ctx_size += n_layer * 1 * dim_embed * dim_embed * ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_layer * 1 * dim_embed * 1 * ggml_type_size(GGML_TYPE_F32);

            // norm1 weight + bias
            ctx_size += 2 * n_layer * dim_embed * ggml_type_size(GGML_TYPE_F32);

            // fc weight + bias
            ctx_size += n_layer * mlp_width * dim_embed * ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_layer * mlp_width * 1 * ggml_type_size(GGML_TYPE_F32);

            // proj weight + bias
            ctx_size += n_layer * mlp_width * dim_embed * ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_layer * mlp_width * 1 * ggml_type_size(GGML_TYPE_F32);

            // norm2 weight + bias
            ctx_size += 2 * n_layer * dim_embed * ggml_type_size(GGML_TYPE_F32);
        }
        ctx_size += (10 + n_layer * 32) * ggml_tensor_overhead();

        printf("%s: ggml tensor size = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));

        return ctx_size;
    }();

    // create the ggml context

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,
        };

        ctx = ggml_init(params);
        if (!ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int32_t n_layer = hparams.n_layer;
        const int32_t dim_embed = hparams.width;
        const int32_t n_head = hparams.n_head;
        const int32_t patch_size = hparams.patch_size;
        const int32_t head_width = hparams.head_width;
        const int32_t img_size = hparams.img_size;
        const int32_t output_dim = hparams.output_dim;
        const int32_t mlp_width = hparams.mlp_ratio * dim_embed;

        const int32_t n_img_emb = hparams.n_img_bed();
        const int32_t grid_size = img_size / patch_size;

        model.visual_enc.transformer_block.resblocks.resize(n_layer);

        model.visual_enc.class_embedding = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);

        model.visual_enc.position_embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim_embed, grid_size * grid_size + 1);

        model.visual_enc.proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output_dim, dim_embed);

        model.visual_enc.conv1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, patch_size, patch_size, 3, dim_embed);

        model.visual_enc.ln_pre_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);
        model.visual_enc.ln_pre_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);

        model.visual_enc.ln_post_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);
        model.visual_enc.ln_post_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);

        model.tensors["class_embedding"]        = model.visual_enc.class_embedding;
        model.tensors["positional_embedding"]   = model.visual_enc.position_embedding;
        model.tensors["proj"]                   = model.visual_enc.proj;
        model.tensors["conv1.weight"]           = model.visual_enc.conv1_w;
        model.tensors["ln_pre.weight"]          = model.visual_enc.ln_pre_w;
        model.tensors["ln_pre.bias"]            = model.visual_enc.ln_pre_b;
        model.tensors["ln_post.weight"]         = model.visual_enc.ln_post_w;
        model.tensors["ln_post.bias"]           = model.visual_enc.ln_post_b;
       
        printf("Done init first %d \n ", mlp_width);
        for (int idx_layer = 0; idx_layer < n_layer; ++idx_layer)
        {
            printf("Done %d layer \n", idx_layer);
            auto &layer = model.visual_enc.transformer_block.resblocks[idx_layer];

            layer.attn.c_attn_attn_w = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, dim_embed, 3 * dim_embed);
            layer.attn.c_attn_attn_b = ggml_new_tensor_1d(
                ctx, GGML_TYPE_F32, 3 * dim_embed);

            layer.attn.c_attn_proj_w = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, dim_embed, dim_embed);
            layer.attn.c_attn_proj_b = ggml_new_tensor_1d(
                ctx, GGML_TYPE_F32, dim_embed);

            layer.ln_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);
            layer.ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);

            layer.fc_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim_embed, mlp_width);
            layer.fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mlp_width);

            layer.proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mlp_width, dim_embed);
            layer.proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);

            layer.ln_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);
            layer.ln_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_embed);
            
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".attn.in_proj_weight"]        = layer.attn.c_attn_attn_w;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".attn.in_proj_bias"]          = layer.attn.c_attn_attn_b;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".attn.out_proj.weight"]       = layer.attn.c_attn_proj_w;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".attn.out_proj.bias"]         = layer.attn.c_attn_proj_b;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".ln_1.weight"]                = layer.ln_1_w;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".ln_1.bias"]                  = layer.ln_1_b;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".mlp.c_fc.weight"]            = layer.fc_w;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".mlp.c_fc.bias"]              = layer.fc_b;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".mlp.c_proj.weight"]          = layer.proj_w;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".mlp.c_proj.bias"]            = layer.proj_b;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".ln_2.weight"]                = layer.ln_2_w;
            model.tensors["transformer.resblocks." + std::to_string(idx_layer) + ".ln_2.bias"]                  = layer.ln_2_b;
            
        }
    }


    // load weights 
    {
        int n_tensors = 0;
        size_t total_size = 0;

        fprintf(stderr, "%s: ", __func__);

        while(true) {

            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[4] = { 1, 1, 1, 1 };

            for (int i=0; i < n_dims; ++i){
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];

            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %d, expected %d\n",
                        __func__, name.data(), (int) nelements, (int) ggml_nelements(tensor));
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2] || tensor->ne[3] != ne[3]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]\n",
                        __func__, name.data(),
                        (int) ne[0], (int) ne[1], (int) ne[2], (int) ne[3],
                        (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2], (int) tensor->ne[3]);
                return false;
            }

            size_t bpe = 0; // bytes per element

            switch (ftype) {
                case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                default:
                        {
                            fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                            return false;
                        }
            };

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), (size_t) nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                fprintf(stderr, ".");
                fflush(stdout);
            }

        }
    
        if (n_tensors != int(model.tensors.size())) {
            fprintf(stderr, "%s: model file has %d tensors, but %d tensors were expected\n", __func__, n_tensors, (int) model.tensors.size());
            return false;
        }

        fprintf(stderr, " done\n");

        fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    
    }

    fin.close();

    return true;
}



int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    visual_clip_params params;
    // params.model = "";

    openclip_visual_model model;
    // sam_state state;
    int64_t t_load_us = 0;

    // if (sam_params_parse(argc, argv, params) == false) {
    //     return 1;
    // }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the image
    // image_u8 img0;
    // if (!sam_image_load_from_file(params.fname_inp, img0)) {
    //     fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
    //     return 1;
    // }
    // fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    // preprocess to f32
    // image_f32 img1;
    // if (!sam_image_preprocess(img0, img1)) {
    //     fprintf(stderr, "%s: failed to preprocess image\n", __func__);
    //     return 1;
    // }
    // fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img1.nx, img1.ny);


    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!open_clip_model_load(params, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

}