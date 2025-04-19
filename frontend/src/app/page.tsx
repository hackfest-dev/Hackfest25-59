"use client"

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const [channelName, setChannelName] = useState()
  const [userName, setUserName] = useState()

  const router = useRouter();

  const onCallClick = () => {
    const query = new URLSearchParams({
      channelName,
      userName
    }).toString()
    router.push('/call?' + query)
  }

  return (
    <div className="flex flex-col w-100 justify-center items-center m-auto p-8 h-screen">
      <p>Welcome to your own video calling app built from scratch!</p>
      <div className='flex flex-col m-4'>
        <label className="block mb-2 text-sm font-medium text-gray-900 dark:text-white" htmlFor='channelName'>Enter your channel name</label>
        <input className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" id='channelName' value={channelName} onChange={(e) => setChannelName(e.target.value)} placeholder='Channel Name' />
      </div>
      <div className='flex flex-col m-4'>
        <label className="block mb-2 text-sm font-medium text-gray-900 dark:text-white" htmlFor='channelName'>Enter your username</label>
        <input className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" id='channelName' value={userName} onChange={(e) => setUserName(e.target.value)} placeholder='Channel Name' />
      </div>
      <button type="button" className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800" onClick={() => { onCallClick() }}>Call</button>

    </div>
  );
}