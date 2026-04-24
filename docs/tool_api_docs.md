# 直播问数智能体接口文档

## 路线查询

**接口地址** `33.69.9.160/agent/driving`

**请求方式** `GET`

**consumes** \`\`

**produces** `["*/*"]`

**接口描述** \`\`

**请求参数**

| 参数名称 | 参数说明 | 请求类型 | 是否必须 | 数据类型 | schema |
| -------- | -------- | -------- | -------- | -------- | ------ |
| end      | end      | query    | true     | string   |        |
| start    | start    | query    | true     | string   |        |

**响应状态**

| 状态码 | 说明         | schema                |
| ------ | ------------ | --------------------- |
| 200    | OK           | Result«DrivingPlanVo» |
| 401    | Unauthorized |                       |
| 403    | Forbidden    |                       |
| 404    | Not Found    |                       |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema         |
| -------- | -------- | -------------- | -------------- |
| code     |          | integer(int32) | integer(int32) |
| data     |          | DrivingPlanVo  | DrivingPlanVo  |
| message  |          | string         |                |

**schema属性说明**

**DrivingPlanVo**

| 参数名称    | 参数说明 | 类型           | schema |
| ----------- | -------- | -------------- | ------ |
| routes      | 路线方案 | array          | Routes |
| routesCount | 路线总数 | integer(int32) |        |

**Routes**

| 参数名称 | 参数说明                             | 类型           | schema   |
| -------- | ------------------------------------ | -------------- | -------- |
| distance | 方案总距离，单位：米                 | integer(int64) |          |
| duration | 方案估算时间（结合路况），单位：分钟 | integer(int64) |          |
| sections |                                      | array          | Sections |
| tags     | 方案标签                             | array          |          |
| toll     | 高速过路费，单位：元                 | integer(int64) |          |

**Sections**

| 参数名称           | 参数说明     | 类型   | schema             |
| ------------------ | ------------ | ------ | ------------------ |
| exitInfos          | 沿途收费站   | array  | ExitInfos          |
| roadName           | 高速名称     | string |                    |
| serviceAreas       | 沿途服务区   | array  | ServiceAreas       |
| trafficCongestions | 沿途拥堵事件 | array  | TrafficCongestions |
| trafficControls    | 沿途管制事件 | array  | TrafficControls    |

**ExitInfos**

| 参数名称       | 参数说明                                                 | 类型           | schema |
| -------------- | -------------------------------------------------------- | -------------- | ------ |
| entranceStatus | 收费站入口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| exportStatus   | 收费站出口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| tollId         | 收费站id                                                 | integer(int32) |        |
| tollName       | 收费站名称                                               | string         |        |

**ServiceAreas**

| 参数名称      | 参数说明                      | 类型           | schema |
| ------------- | ----------------------------- | -------------- | ------ |
| directionType | 方向，00 双向，01上行，02下行 | string         |        |
| latitude      | 纬度                          | number(double) |        |
| longitude     | 经度                          | number(double) |        |
| roadId        | 高速id                        | integer(int32) |        |
| serviceId     | 服务区id                      | integer(int64) |        |
| serviceName   | 服务区名称                    | string         |        |

**TrafficCongestions**

| 参数名称        | 参数说明                                      | 类型           | schema |
| --------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone  | 开始桩号                                      | integer(int32) |        |
| beginTime       | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures | 管制说明                                      | string         |        |
| des             | 事件描述                                      | string         |        |
| directionType   | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone    | 结束桩号                                      | integer(int32) |        |
| eventType       | 事件大类编码                                  | string         |        |
| id              | 事件id                                        | string         |        |
| roadAmbleMile   | 缓行公里数                                    | number(double) |        |
| roadId          | 高速id                                        | integer(int32) |        |
| subEventType    | 事件小类编码                                  | string         |        |

**TrafficControls**

| 参数名称        | 参数说明                                      | 类型           | schema |
| --------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone  | 开始桩号                                      | integer(int32) |        |
| beginTime       | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures | 管制说明                                      | string         |        |
| des             | 事件描述                                      | string         |        |
| directionType   | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone    | 结束桩号                                      | integer(int32) |        |
| eventType       | 事件大类编码                                  | string         |        |
| id              | 事件id                                        | string         |        |
| roadAmbleMile   | 缓行公里数                                    | number(double) |        |
| roadId          | 高速id                                        | integer(int32) |        |
| subEventType    | 事件小类编码                                  | string         |        |

**响应示例**

```json
{
  "code": 0,
  "data": {
    "routes": [
      {
        "distance": 0,
        "duration": 0,
        "sections": [
          {
            "exitInfos": [
              {
                "entranceStatus": 0,
                "exportStatus": 0,
                "tollId": 0,
                "tollName": ""
              }
            ],
            "roadName": "",
            "serviceAreas": [
              {
                "directionType": "",
                "latitude": 0,
                "longitude": 0,
                "roadId": 0,
                "serviceId": 0,
                "serviceName": ""
              }
            ],
            "trafficCongestions": [
              {
                "beginMilestone": 0,
                "beginTime": "",
                "controlMeasures": "",
                "des": "",
                "directionType": "",
                "endMilestone": 0,
                "eventType": "",
                "id": "",
                "roadAmbleMile": 0,
                "roadId": 0,
                "subEventType": ""
              }
            ],
            "trafficControls": [
              {
                "beginMilestone": 0,
                "beginTime": "",
                "controlMeasures": "",
                "des": "",
                "directionType": "",
                "endMilestone": 0,
                "eventType": "",
                "id": "",
                "roadAmbleMile": 0,
                "roadId": 0,
                "subEventType": ""
              }
            ]
          }
        ],

        "tags": [],

        "toll": 0
      }
    ],
    "routesCount": 0
  },
  "message": ""
}
```

## 路况查询

**接口地址** `33.69.3.160:8081/agent/event`

**请求方式** `GET`

**consumes** \`\`

**produces** `["*/*"]`

**接口描述** \`\`

**请求参数**

| 参数名称 | 参数说明 | 请求类型 | 是否必须 | 数据类型 | schema |
| -------- | -------- | -------- | -------- | -------- | ------ |
| road     | road     | query    | true     | string   |        |

**响应状态**

| 状态码 | 说明         | schema                        |
| ------ | ------------ | ----------------------------- |
| 200    | OK           | Result«List«RoadConditionVo»» |
| 401    | Unauthorized |                               |
| 403    | Forbidden    |                               |
| 404    | Not Found    |                               |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema          |
| -------- | -------- | -------------- | --------------- |
| code     |          | integer(int32) | integer(int32)  |
| data     |          | array          | RoadConditionVo |
| message  |          | string         |                 |

**schema属性说明**

**RoadConditionVo**

| 参数名称           | 参数说明     | 类型           | schema         |
| ------------------ | ------------ | -------------- | -------------- |
| congestionInfoList | 拥堵情况     | array          | CongestionInfo |
| exitInfoList       | 出⼝情况     | array          | ExitInfo       |
| roadGbCode         | 高速编号     | string         |                |
| roadId             | 高速id       | integer(int32) |                |
| roadName           | 高速名称     | string         |                |
| serviceAreaList    | 服务区情况   | array          | ServiceArea    |
| trafficControlList | 交通管制情况 | array          | TrafficControl |

**CongestionInfo**

| 参数名称        | 参数说明                                      | 类型           | schema |
| --------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone  | 开始桩号                                      | integer(int32) |        |
| beginTime       | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures | 管制说明                                      | string         |        |
| des             | 事件描述                                      | string         |        |
| directionType   | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone    | 结束桩号                                      | integer(int32) |        |
| eventType       | 事件大类编码                                  | string         |        |
| expectedEndTime | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| id              | 事件id                                        | string         |        |
| roadAmbleMile   | 缓行公里数                                    | number(double) |        |
| roadId          | 高速id                                        | integer(int32) |        |
| subEventType    | 事件小类编码                                  | string         |        |

**ExitInfo**

| 参数名称       | 参数说明                                                 | 类型           | schema |
| -------------- | -------------------------------------------------------- | -------------- | ------ |
| entranceStatus | 收费站入口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| exportStatus   | 收费站出口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| tollId         | 收费站id                                                 | integer(int32) |        |
| tollName       | 收费站名称                                               | string         |        |

**ServiceArea**

| 参数名称      | 参数说明                      | 类型           | schema |
| ------------- | ----------------------------- | -------------- | ------ |
| directionType | 方向，00 双向，01上行，02下行 | string         |        |
| serviceId     | 服务区id                      | integer(int64) |        |
| serviceName   | 服务区名称                    | string         |        |
| statusTag     | 服务区拥挤状态                | string         |        |

**TrafficControl**

| 参数名称        | 参数说明                                      | 类型           | schema |
| --------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone  | 开始桩号                                      | integer(int32) |        |
| beginTime       | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures | 管制说明                                      | string         |        |
| des             | 事件描述                                      | string         |        |
| directionType   | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone    | 结束桩号                                      | integer(int32) |        |
| eventType       | 事件大类编码                                  | string         |        |
| expectedEndTime | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| id              | 事件id                                        | string         |        |
| roadAmbleMile   | 缓行公里数                                    | number(double) |        |
| roadId          | 高速id                                        | integer(int32) |        |
| subEventType    | 事件小类编码                                  | string         |        |

**响应示例**

```json
{
  "code": 0,
  "data": [
    {
      "congestionInfoList": [
        {
          "beginMilestone": 0,
          "beginTime": "",
          "controlMeasures": "",
          "des": "",
          "directionType": "",
          "endMilestone": 0,
          "eventType": "",
          "expectedEndTime": "",
          "id": "",
          "roadAmbleMile": 0,
          "roadId": 0,
          "subEventType": ""
        }
      ],
      "exitInfoList": [
        {
          "entranceStatus": 0,
          "exportStatus": 0,
          "tollId": 0,
          "tollName": ""
        }
      ],
      "roadGbCode": "",
      "roadId": 0,
      "roadName": "",
      "serviceAreaList": [
        {
          "directionType": "",
          "serviceId": 0,
          "serviceName": "",
          "statusTag": ""
        }
      ],
      "trafficControlList": [
        {
          "beginMilestone": 0,
          "beginTime": "",
          "controlMeasures": "",
          "des": "",
          "directionType": "",
          "endMilestone": 0,
          "eventType": "",
          "expectedEndTime": "",
          "id": "",
          "roadAmbleMile": 0,
          "roadId": 0,
          "subEventType": ""
        }
      ]
    }
  ],
  "message": ""
}
```

## 服务区查询

**接口地址** `33.69.3.160:8081/agent/service`

**请求方式** `GET`

**consumes** \`\`

**produces** `["*/*"]`

**接口描述** \`\`

**请求参数**

| 参数名称 | 参数说明 | 请求类型 | 是否必须 | 数据类型 | schema |
| -------- | -------- | -------- | -------- | -------- | ------ |
| keyword  | keyword  | query    | true     | string   |        |

**响应状态**

| 状态码 | 说明         | schema                      |
| ------ | ------------ | --------------------------- |
| 200    | OK           | Result«List«ServiceInfoVo»» |
| 401    | Unauthorized |                             |
| 403    | Forbidden    |                             |
| 404    | Not Found    |                             |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema         |
| -------- | -------- | -------------- | -------------- |
| code     |          | integer(int32) | integer(int32) |
| data     |          | array          | ServiceInfoVo  |
| message  |          | string         |                |

**schema属性说明**

**ServiceInfoVo**

| 参数名称       | 参数说明                        | 类型           | schema   |
| -------------- | ------------------------------- | -------------- | -------- |
| chargeList     | 充电桩数据集合                  | array          | ChargeVO |
| commercialList | 商业服务数据集合                | array          | StoreVo  |
| direction      | 服务区方向                      | string         |          |
| directionName  | 服务区方向名称                  | string         |          |
| directionType  | 方向，00 双向，01 上行，02 下行 | string         |          |
| latitude       | 服务区纬度                      | number(double) |          |
| longitude      | 服务区经度                      | number(double) |          |
| milestone      | 桩号                            | string         |          |
| milestoneNum   | 桩号（数字）                    | number(double) |          |
| roadGbCode     | 高速编号                        | string         |          |
| roadId         | 高速id                          | integer(int32) |          |
| roadName       | 高速名称                        | string         |          |
| serviceId      | 服务区分区id                    | integer(int64) |          |
| serviceName    | 服务区名称                      | string         |          |
| statusTag      | 服务区拥挤状态                  | string         |          |
| tags           | 其他配套设施                    | array          |          |

**ChargeVO**

| 参数名称               | 参数说明           | 类型           | schema |
| ---------------------- | ------------------ | -------------- | ------ |
| manufacturerLogo       | 充电品牌url        | string         |        |
| manufacturerName       | 充电品牌           | string         |        |
| totalACChargingNum     | 慢充充电桩总数     | integer(int32) |        |
| totalChargingNum       | 充电桩总数         | integer(int32) |        |
| totalDCChargingNum     | 快充充电桩总数     | integer(int32) |        |
| totalFreeACChargingNum | 空闲慢充充电桩总数 | integer(int32) |        |
| totalFreeChargingNum   | 空闲充电桩总数     | integer(int32) |        |
| totalFreeDCChargingNum | 空闲快充充电桩总数 | integer(int32) |        |

**StoreVo**

| 参数名称          | 参数说明     | 类型   | schema |
| ----------------- | ------------ | ------ | ------ |
| businessEndTime   | 营业结束时间 | string |        |
| businessStartTime | 营业开始时间 | string |        |
| code              | 店铺编码     | string |        |
| name              | 店铺名称     | string |        |

**响应示例**

```json
{
  "code": 0,
  "data": [
    {
      "chargeList": [
        {
          "manufacturerLogo": "",
          "manufacturerName": "",
          "totalACChargingNum": 0,
          "totalChargingNum": 0,
          "totalDCChargingNum": 0,
          "totalFreeACChargingNum": 0,
          "totalFreeChargingNum": 0,
          "totalFreeDCChargingNum": 0
        }
      ],
      "commercialList": [
        {
          "businessEndTime": "",
          "businessStartTime": "",
          "code": "",
          "name": ""
        }
      ],
      "direction": "",
      "directionName": "",
      "directionType": "",
      "latitude": 0,
      "longitude": 0,
      "milestone": "",
      "milestoneNum": 0,
      "roadGbCode": "",
      "roadId": 0,
      "roadName": "",
      "serviceId": 0,
      "serviceName": "",
      "statusTag": "",

      "tags": []
    }
  ],
  "message": ""
}
```

## 整体⾼速路情况

**接口地址** `33.69.3.160/agent/topN`

**请求方式** `GET`

**consumes** \`\`

**produces** `["*/*"]`

**接口描述** \`\`

**请求参数**

暂无

**响应状态**

| 状态码 | 说明         | schema                |
| ------ | ------------ | --------------------- |
| 200    | OK           | Result«IncidentTopVO» |
| 401    | Unauthorized |                       |
| 403    | Forbidden    |                       |
| 404    | Not Found    |                       |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema         |
| -------- | -------- | -------------- | -------------- |
| code     |          | integer(int32) | integer(int32) |
| data     |          | IncidentTopVO  | IncidentTopVO  |
| message  |          | string         |                |

**schema属性说明**

**IncidentTopVO**

| 参数名称       | 参数说明                                  | 类型   | schema |
| -------------- | ----------------------------------------- | ------ | ------ |
| congestionTopN | 拥堵汇总                                  | array  |        |
| controlTopN    | 主线管制汇总                              | array  |        |
| exitTopN       | 收费站管制汇总                            | array  |        |
| queryTime      | 查询时间，时间格式为: yyyy-MM-dd HH:mm:ss | string |        |

**地图返回事件数据对象**

| 参数名称        | 参数说明                                                                                                                     | 类型           | schema |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------- | ------ |
| beginMilestone  | 开始桩号                                                                                                                     | integer(int32) |        |
| beginTime       | 事件开始时间                                                                                                                 | string         |        |
| cameraFirst     | 视频是否优先展示 true优先展示视频                                                                                            | boolean        |        |
| cameraIndexCode | 视频监控点编号                                                                                                               | string         |        |
| controlMeasures | 管制措施                                                                                                                     | string         |        |
| des             | 事件描述                                                                                                                     | string         |        |
| directionType   | 方向，00 无，01上行，02下行                                                                                                  | string         |        |
| endMilestone    | 结束桩号                                                                                                                     | integer(int32) |        |
| eventClass      | v1.1.0版本，事件归属编码。01:站点管制, 02:主线管制, 03:道路缓行, 04:交通事故, 05:道路施工, 06:路面状况, 07:车辆故障, 08:其他 | string         |        |
| eventType       | v1.1.0版本, 事件大类查看eventType类型表                                                                                      | string         |        |
| expectedTime    | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss                                                                                | string         |        |
| id              | 事件id                                                                                                                       | string         |        |
| jeeves          | 占道情况                                                                                                                     | string         |        |
| latitude        | 事件发生位置纬度                                                                                                             | string         |        |
| longitude       | 事件发生位置经度                                                                                                             | string         |        |
| pictureUrl      | 事件图片地址                                                                                                                 | string         |        |
| rescueStatus    | 救援进度. -1:没有救援进度, 0:救援开始, 1:救援中, 2:救援结束                                                                  | integer(int32) |        |
| road            | 高速id                                                                                                                       | integer(int32) |        |
| roadAmbleMile   | 缓行公里数                                                                                                                   | number(double) |        |
| roadGBCode      | 高速编码                                                                                                                     | string         |        |
| roadName        | 高速名称                                                                                                                     | string         |        |
| situationRemark | 现场情况备注                                                                                                                 | string         |        |
| subEventType    | 事件小类编码.                                                                                                                | string         |        |
| subEventTypeId  | 事件小类ID                                                                                                                   | string         |        |

**CongestionSum**

| 参数名称  | 参数说明   | 类型           | schema |
| --------- | ---------- | -------------- | ------ |
| totalMile | 拥堵总里程 | number(double) |        |

**TollControlDTO**

| 参数名称             | 参数说明                                                                                                                                                                                                                           | 类型           | schema |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------ |
| controlType          | 站点管制：<br>管制类型id，10202关闭，10203限流，10204分流<br>主线管制：<br>管制类型id，10101封闭部分车道，10102单向封道，10103双向封道，10105借道通行，10106卡口，10107限速，10108主线互通（匝道），10109主线限流，10110硬路肩开放 | integer(int32) |        |
| controlTypeName      | 管制类型名称                                                                                                                                                                                                                       | string         |        |
| delFlag              | 是否删除0：未删除1：已删除                                                                                                                                                                                                         | integer(int32) |        |
| des                  | 管制措施说明                                                                                                                                                                                                                       | string         |        |
| direction            | 方向, 100700上行，100701下行                                                                                                                                                                                                       | integer(int32) |        |
| directionName        | 方向名称                                                                                                                                                                                                                           | string         |        |
| endTime              | 结束时间，格式:YYYY-MM-dd HH:mm:ss。当前时间大于此时间时，说明管制结束                                                                                                                                                             | string         |        |
| entrance             | 出入口：0 出口 1 入口                                                                                                                                                                                                              | integer(int32) |        |
| entranceName         | 出入口名称                                                                                                                                                                                                                         | string         |        |
| eventId              | 关联事件id                                                                                                                                                                                                                         | string         |        |
| id                   | 管制措施id                                                                                                                                                                                                                         | string         |        |
| limitMeasureTypeName | 限流管制措施详细类型                                                                                                                                                                                                               | string         |        |
| messageId            |                                                                                                                                                                                                                                    | string         |        |
| roadGBCode           | 路段国标号                                                                                                                                                                                                                         | string         |        |
| roadId               | 路段ID                                                                                                                                                                                                                             | string         |        |
| roadName             | 路段名称                                                                                                                                                                                                                           | string         |        |
| startTime            | 开始时间，格式:YYYY-MM-dd HH:mm:ss                                                                                                                                                                                                 | string         |        |
| timestamp            | 消息发布时间的时间戳                                                                                                                                                                                                               | integer(int64) |        |
| tollId               | 收费站id                                                                                                                                                                                                                           | integer(int32) |        |
| tollName             | 收费站名称                                                                                                                                                                                                                         | string         |        |

**响应示例**

```json
{
  "code": 0,
  "data": {
    "congestionTopN": [
      {
        "beginMilestone": 0,
        "beginTime": "",
        "cameraFirst": true,
        "cameraIndexCode": "",
        "controlMeasures": "",
        "des": "",
        "directionType": "",
        "endMilestone": 0,
        "eventClass": "",
        "eventType": "",
        "expectedTime": "",
        "id": "",
        "jeeves": "",
        "latitude": "",
        "longitude": "",
        "pictureUrl": "",
        "rescueStatus": 0,
        "road": 0,
        "roadAmbleMile": 0,
        "roadGBCode": "",
        "roadName": "",
        "situationRemark": "",
        "subEventType": "",
        "subEventTypeId": ""
      }
    ],
    "controlTopN": [
      {
        "controlType": 0,
        "controlTypeName": "",
        "delFlag": 0,
        "des": "",
        "direction": 0,
        "directionName": "",
        "endTime": "",
        "entrance": 0,
        "entranceName": "",
        "eventId": "",
        "id": "",
        "limitMeasureTypeName": "",
        "messageId": "",
        "roadGBCode": "",
        "roadId": "",
        "roadName": "",
        "startTime": "",
        "timestamp": 0,
        "tollId": 0,
        "tollName": ""
      }
    ],
    "exitTopN": [
      {
        "controlType": 0,
        "controlTypeName": "",
        "delFlag": 0,
        "des": "",
        "direction": 0,
        "directionName": "",
        "endTime": "",
        "entrance": 0,
        "entranceName": "",
        "eventId": "",
        "id": "",
        "limitMeasureTypeName": "",
        "messageId": "",
        "roadGBCode": "",
        "roadId": "",
        "roadName": "",
        "startTime": "",
        "timestamp": 0,
        "tollId": 0,
        "tollName": ""
      }
    ],
    "queryTime": ""
  },
  "message": ""
}
```

EVENT_TYPE 映射关系

| 事件大类编码 | 事件大类       |
| ------------ | -------------- |
| 01           | 交通事件       |
| 02           | 交通灾害       |
| 03           | 交通气象       |
| 04           | 路面状况       |
| 05           | 路面施工       |
| 06           | 活动           |
| 07           | 重大事件       |
| 09           | 其他           |
| 97           | 车辆故障       |
| 98           | 服务区事件     |
| 99           | 收费站入口关闭 |
| 100          | 收费站入口限流 |
| 101          | 收费站出口关闭 |
| 104          | 收费站出口分流 |
| 103          | 主线管制       |
| 105          | 道路缓行       |
