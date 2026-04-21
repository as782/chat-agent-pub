## 路线查询

**接口地址** `33.69.3.160:8081/agent/driving`

**请求方式** `GET`

**consumes** ``

**produces** `["*/*"]`

**接口描述** ``

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
| 404    | Not Found    |                       |

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

| 参数名称        | 参数说明     | 类型   | schema          |
| --------------- | ------------ | ------ | --------------- |
| roadName        | 高速名称     | string |                 |
| serviceAreas    | 沿途服务区   | array  | ServiceAreas    |
| trafficControls | 沿途管制事件 | array  | TrafficControls |

**ServiceAreas**

| 参数名称      | 参数说明                      | 类型           | schema |
| ------------- | ----------------------------- | -------------- | ------ |
| directionType | 方向，00 双向，01上行，02下行 | string         |        |
| latitude      | 纬度                          | number(double) |        |
| longitude     | 经度                          | number(double) |        |
| roadId        | 高速id                        | integer(int32) |        |
| serviceId     | 服务区id                      | integer(int64) |        |
| serviceName   | 服务区名称                    | string         |        |

**TrafficControls**

| 参数名称        | 参数说明                                      | 类型           | schema |
| --------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone  | 开始桩号                                      | integer(int32) |        |
| beginTime       | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures | 管制说明                                      | string         |        |
| des             | 事件描述                                      | string         |        |
| directionType   | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone    | 结束桩号                                      | integer(int32) |        |
| eventType       | 事件大类编码                                  | string         |        |
| expectedEndTime | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
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
            "trafficControls": [
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

**接口地址** `33.69.3.160:8081/agent/event`

**请求方式** `GET`

**consumes** ``

**produces** `["*/*"]`

**接口描述** ``

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
| 404    | Not Found    |                               |

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
| beginTime       | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures | 管制说明                                      | string         |        |
| des             | 事件描述                                      | string         |        |
| directionType   | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone    | 结束桩号                                      | integer(int32) |        |
| eventType       | 事件大类编码                                  | string         |        |
| expectedEndTime | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| id              | 事件id                                        | string         |        |
| roadAmbleMile   | 缓行公里数                                    | number(double) |        |
| roadId          | 高速id                                        | integer(int32) |        |
| subEventType    | 事件小类编码                                  | string         |        |

**ExitInfo**

| 参数名称       | 参数说明                                                 | 类型           | schema |
| -------------- | -------------------------------------------------------- | -------------- | ------ |
| entranceStatus | 收费站入口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| exportStatus   | 收费站出口状态。0: 开启，10202关闭，10203限流，10204分流 | integer(int32) |        |
| tollId         | 收费站id                                                 | integer(int32) |        |
| tollName       | 收费站名称                                               | string         |        |

**ServiceArea**

| 参数名称      | 参数说明                      | 类型           | schema |
| ------------- | ----------------------------- | -------------- | ------ |
| directionType | 方向，00 双向，01上行，02下行 | string         |        |
| serviceId     | 服务区id                      | integer(int64) |        |
| serviceName   | 服务区名称                    | string         |        |
| statusTag     | 服务区拥挤状态                | string         |        |

**TrafficControl**

| 参数名称        | 参数说明                                      | 类型           | schema |
| --------------- | --------------------------------------------- | -------------- | ------ |
| beginMilestone  | 开始桩号                                      | integer(int32) |        |
| beginTime       | 事件开始时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
| controlMeasures | 管制说明                                      | string         |        |
| des             | 事件描述                                      | string         |        |
| directionType   | 方向，00 双向，01上行，02下行                 | string         |        |
| endMilestone    | 结束桩号                                      | integer(int32) |        |
| eventType       | 事件大类编码                                  | string         |        |
| expectedEndTime | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss | string         |        |
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

**接口地址** `33.69.3.160:8081/agent/service`

**请求方式** `GET`

**consumes** ``

**produces** `["*/*"]`

**接口描述** ``

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
| 404    | Not Found    |                             |

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
| directionType  | 方向，00 双向，01 上行，02 下行 | string         |          |
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

**接口地址** `33.69.3.160:8081/agent/topN`

**请求方式** `GET`

**consumes** ``

**produces** `["*/*"]`

**接口描述** ``

**请求参数**

暂无

**响应状态**

| 状态码 | 说明         | schema                |
| ------ | ------------ | --------------------- |
| 200    | OK           | Result«IncidentTopVO» |
| 401    | Unauthorized |                       |
| 403    | Forbidden    |                       |
| 404    | Not Found    |                       |

**响应参数**

| 参数名称 | 参数说明 | 类型           | schema         |
| -------- | -------- | -------------- | -------------- |
| code     |          | integer(int32) | integer(int32) |
| data     |          | IncidentTopVO  | IncidentTopVO  |
| message  |          | string         |                |

**schema属性说明**

**IncidentTopVO**

| 参数名称       | 参数说明                                  | 类型          | schema               |
| -------------- | ----------------------------------------- | ------------- | -------------------- |
| accidentTopN   | 事故汇总                                  | array         | 地图返回事件数据对象 |
| congestion     | 拥堵                                      | CongestionSum | CongestionSum        |
| congestionTopN | 拥堵汇总                                  | array         | 地图返回事件数据对象 |
| controlTopN    | 管制汇总                                  | array         | 地图返回事件数据对象 |
| queryTime      | 查询事件，事件格式为: yyyy-MM-dd HH:mm:ss | string        |                      |

**地图返回事件数据对象**

| 参数名称        | 参数说明                                                                                                                     | 类型           | schema |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------- | ------ |
| beginMilestone  | 开始桩号                                                                                                                     | integer(int32) |        |
| beginTime       | 事件开始时间                                                                                                                 | string         |        |
| controlMeasures | 管制措施                                                                                                                     | string         |        |
| des             | 事件描述                                                                                                                     | string         |        |
| directionType   | 方向，00 无，01上行，02下行                                                                                                  | string         |        |
| endMilestone    | 结束桩号                                                                                                                     | integer(int32) |        |
| eventClass      | v1.1.0版本，事件归属编码。01:站点管制, 02:主线管制, 03:道路缓行, 04:交通事故, 05:道路施工, 06:路面状况, 07:车辆故障, 08:其他 | string         |        |
| eventType       | v1.1.0版本, 事件大类查看eventType类型表                                                                                      | string         |        |
| expectedTime    | 预计结束时间. 事件格式为: yyyy-MM-dd HH:mm:ss                                                                                | string         |        |
| id              | 事件id                                                                                                                       | string         |        |
| jeeves          | 占道情况                                                                                                                     | string         |        |
| latitude        | 事件发生位置纬度                                                                                                             | string         |        |
| longitude       | 事件发生位置经度                                                                                                             | string         |        |
| road            | 高速id                                                                                                                       | integer(int32) |        |
| roadAmbleMile   | 缓行公里数                                                                                                                   | number(double) |        |
| roadGBCode      |                                                                                                                              | string         |        |
| roadName        | 高速名称                                                                                                                     | string         |        |
| situationRemark | 现场情况备注                                                                                                                 | string         |        |
| subEventType    | 事件小类编码.                                                                                                                | string         |        |
| subEventTypeId  | 事件小类ID                                                                                                                   | string         |        |

**CongestionSum**

| 参数名称  | 参数说明   | 类型           | schema |
| --------- | ---------- | -------------- | ------ |
| totalMile | 拥堵总里程 | number(double) |        |

**响应示例**

```json
{
  "code": 0,
  "data": {
    "accidentTopN": [
      {
        "beginMilestone": 0,
        "beginTime": "",
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
        "road": 0,
        "roadAmbleMile": 0,
        "roadGBCode": "",
        "roadName": "",
        "situationRemark": "",
        "subEventType": "",
        "subEventTypeId": ""
      }
    ],
    "congestion": {
      "totalMile": 0
    },
    "congestionTopN": [
      {
        "beginMilestone": 0,
        "beginTime": "",
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
        "beginMilestone": 0,
        "beginTime": "",
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
        "road": 0,
        "roadAmbleMile": 0,
        "roadGBCode": "",
        "roadName": "",
        "situationRemark": "",
        "subEventType": "",
        "subEventTypeId": ""
      }
    ],
    "queryTime": ""
  },
  "message": ""
}
```
