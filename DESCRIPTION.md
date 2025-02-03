# Welcome to Jebosio API documenation !
<>
# Jebosio API reference
  #### version: 0.0.0
  #### Servers: http://203.252.147.202:8000
  - Sever spec : 
  | CPU | RAM | GPU | SSD |
  | --- | --- | --- | --- |
  | AMD Ryzen 5950X 16core / 32threads | 128GB | RTX 3090 24GB(2EA)	| M.2 1TB |


  <!-- ### Summary
| request parameters | Descriptions | Request parameters |  
| --- | --- | --- |
| /api/v1.0/img-kpt | measure the cloth size  | cloth image(jpg), depth image(jpg) |
| 1200 x 800 | 1 MP | 500 images per minute |
| 1600 x 1200 | 2 MP | 500 / 2 = 250 images per minute |
| 2500 x 1600 | 4 MP | 500 / 4 = 125 images per minute |
| 4000 x 2500 | 10 MP | 500 / 10 = 50 images per minute |
| 6250 x 4000 | 25 MP | 500 / 25 = 20 images per minute |
     -->

<!-- 

## 1. Remove Background

Explore our API doucumentation and try your application or workflow!

It is based on U-net to remove background
<p>
  <img src="https://raw.githubusercontent.com/tae-uk0403/Jebosio/main/pants.png" width="300">
  <img src="https://raw.githubusercontent.com/tae-uk0403/Jebosio/main/40.png" width="300">
</p>































## Output formats

You can request one of three formats via the `format` parameter:

| Format | Resolution | Pros and cons | Example |
| --- | --- | --- | --- |
| PNG | Up to 10 Megapixelse.g. 4000x2500 | + Simple integration+ Supports transparency- Large file size | https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-53909f9ef9d8156ec0d4e7dc67fec610430d489b1298fd2acbf2f792eadc9a7e.pnghttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-53909f9ef9d8156ec0d4e7dc67fec610430d489b1298fd2acbf2f792eadc9a7e.pnghttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-53909f9ef9d8156ec0d4e7dc67fec610430d489b1298fd2acbf2f792eadc9a7e.png |
| JPG | Up to 25 Megapixelse.g. 6250x4000 | + Simple Integration+ Small file size- No transparency supported | https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-51eb3b4000500e3a227c5a6a3ab50261857cdce218a45aa347b55c3e1999e9fb.jpghttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-51eb3b4000500e3a227c5a6a3ab50261857cdce218a45aa347b55c3e1999e9fb.jpghttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-51eb3b4000500e3a227c5a6a3ab50261857cdce218a45aa347b55c3e1999e9fb.jpg |
| ZIP | Up to 25 Megapixelse.g. 6250x4000 | + Small file size+ Supports transparency- Integration requires compositing | https://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-671d12c94040cfb91e6dd128db2d2ae73eddd03595f297ad022d3ae57d7f39d9.ziphttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-671d12c94040cfb91e6dd128db2d2ae73eddd03595f297ad022d3ae57d7f39d9.ziphttps://static.remove.bg/remove-bg-web/64a0683dccf7a5ae4c79ca1c1a49b5fd8a115f1d/assets/api-docs/example-tiger-671d12c94040cfb91e6dd128db2d2ae73eddd03595f297ad022d3ae57d7f39d9.zip |

Please note that **PNG images above 10 megapixels are not supported**. If you require transparency for images of that size, use the ZIP format (see below). If you don't need transparency (e.g. white background), we recommend JPG.



## Rate Limit

You can process up to **500 images per minute** through the API, depending on the input image resolution in megapixels.

Examples:

| Input image | Megapixels | Effective Rate Limit |
| --- | --- | --- |
| 625 x 400 | 1 MP | 500 images per minute |
| 1200 x 800 | 1 MP | 500 images per minute |
| 1600 x 1200 | 2 MP | 500 / 2 = 250 images per minute |
| 2500 x 1600 | 4 MP | 500 / 4 = 125 images per minute |
| 4000 x 2500 | 10 MP | 500 / 10 = 50 images per minute |
| 6250 x 4000 | 25 MP | 500 / 25 = 20 images per minute |

Exceed of rate limits leads to a HTTP status 429 response (no credits charged). Clients can use the following response headers to gracefully handle rate limits:

| Response header | Sample value | Description |
| --- | --- | --- |
| X-RateLimit-Limit | 500 | Total rate limit in megapixel images |
| X-RateLimit-Remaining | 499 | Remaining rate limit for this minute |
| X-RateLimit-Reset | 1696915907 | Unix timestamp when rate limit will reset |
| Retry-After | 59 | Seconds until rate limit will reset (only present if rate limit exceeded) |

Higher Rate Limits are available [upon request](https://www.remove.bg/support/contact?subject=Rate+Limit+Requirements).


## API Change-log

Most recent API updates:


- **2024-04-16:** Making a page
- **2021-12-07:** Added foreground position and size to background removal responses. (JSON fields: `foreground_top`, `foreground_left`, -->